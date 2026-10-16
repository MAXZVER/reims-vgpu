//! Descriptors owned by translation rather than by a guest argument table.
//!
//! A retained `air.get_null_texture_2d()` has a real sampled-image descriptor
//! in the executable interface. Its synthetic index is not a Metal texture
//! slot. Provision only the reflected shape this rail can prove it represents.

use crate::backend::vulkan::engine::{SampledByteOrigin, SampledImageResource, SampledSource};
use crate::observe::Decline;
use crate::runtime::spirv_bind::{
    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET, SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE,
};
use metal2vulkan::meta::{TextureComponent, TextureDimension};
use metal2vulkan::reflect::{
    ResourceAccess, ResourceKind, ShaderReflection, RESOURCE_DESCRIPTOR_SET,
};
use std::sync::{Arc, OnceLock};

/// A distinct contract check preventing a synthetic image from being bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SynthesizedImageDecline {
    DescriptorMissing { synthetic_index: u32 },
    DescriptorSet { binding: u32, set: u32 },
    DescriptorCount { binding: u32, count: u32 },
    BindingOutsideTextureBand { binding: u32 },
    GuestParameter { binding: u32 },
    Access { binding: u32 },
    ShapeMissing { binding: u32 },
    ShapeUnsupported { binding: u32 },
    BindingOverflow { binding: u32 },
    BindingOccupied { binding: u32 },
}

impl Decline for SynthesizedImageDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::DescriptorMissing { .. } => "draw_synth_null_descriptor_missing",
            Self::DescriptorSet { .. } => "draw_synth_null_descriptor_set",
            Self::DescriptorCount { .. } => "draw_synth_null_descriptor_count",
            Self::BindingOutsideTextureBand { .. } => {
                "draw_synth_null_binding_outside_texture_band"
            }
            Self::GuestParameter { .. } => "draw_synth_null_has_guest_parameter",
            Self::Access { .. } => "draw_synth_null_access_unsupported",
            Self::ShapeMissing { .. } => "draw_synth_null_shape_missing",
            Self::ShapeUnsupported { .. } => "draw_synth_null_shape_unsupported",
            Self::BindingOverflow { .. } => "draw_synth_null_binding_overflow",
            Self::BindingOccupied { .. } => "draw_synth_null_binding_occupied",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::DescriptorMissing { synthetic_index } => {
                vec![("synthetic_index", synthetic_index.to_string())]
            }
            Self::DescriptorSet { binding, set } => {
                vec![("binding", binding.to_string()), ("set", set.to_string())]
            }
            Self::DescriptorCount { binding, count } => {
                vec![
                    ("binding", binding.to_string()),
                    ("count", count.to_string()),
                ]
            }
            Self::BindingOutsideTextureBand { binding }
            | Self::GuestParameter { binding }
            | Self::Access { binding }
            | Self::ShapeMissing { binding }
            | Self::ShapeUnsupported { binding }
            | Self::BindingOverflow { binding }
            | Self::BindingOccupied { binding } => vec![("binding", binding.to_string())],
        }
    }
}

/// Add retained translator-owned null images using final reflection, with the
/// same fragment relocation as the executable variant. Ordinary guest textures
/// never enter this path. Unsupported synthetic shapes retain a typed refusal.
pub fn append_null_textures(
    reflection: &ShaderReflection,
    fragment_relocated: bool,
    images: &mut Vec<SampledImageResource>,
) -> Result<(), SynthesizedImageDecline> {
    use SynthesizedImageDecline as Refusal;

    for resource in &reflection.bindings {
        if resource.kind != ResourceKind::SynthesizedNullTexture {
            continue;
        }
        let descriptor = resource.descriptor.ok_or(Refusal::DescriptorMissing {
            synthetic_index: resource.metal_index,
        })?;
        let binding = descriptor.binding;
        if descriptor.set != RESOURCE_DESCRIPTOR_SET {
            return Err(Refusal::DescriptorSet {
                binding,
                set: descriptor.set,
            });
        }
        if descriptor.count != 1 {
            return Err(Refusal::DescriptorCount {
                binding,
                count: descriptor.count,
            });
        }
        if !(TEXTURE_BINDING_BASE..SAMPLER_BINDING_BASE).contains(&binding) {
            return Err(Refusal::BindingOutsideTextureBand { binding });
        }
        if resource.param_index.is_some() {
            return Err(Refusal::GuestParameter { binding });
        }
        if resource.access != Some(ResourceAccess::Sampled) {
            return Err(Refusal::Access { binding });
        }
        let shape = resource
            .texture_shape
            .as_ref()
            .ok_or(Refusal::ShapeMissing { binding })?;
        if shape.dimension != TextureDimension::D2
            || shape.component != TextureComponent::Float
            || shape.arrayed
            || shape.multisampled
            || shape.writable
            || shape.array_ref
            || shape.array_length.is_some()
            || shape.storage_format.is_some()
        {
            return Err(Refusal::ShapeUnsupported { binding });
        }
        let binding = binding
            .checked_add(if fragment_relocated {
                FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
            } else {
                0
            })
            .ok_or(Refusal::BindingOverflow { binding })?;
        if images.iter().any(|image| image.binding == binding) {
            return Err(Refusal::BindingOccupied { binding });
        }
        // A real initialized image keeps the Vulkan 1.2 path valid without an
        // optional nullDescriptor feature. No guest allocation or content is
        // replaced: reflection explicitly identifies this as translator-owned.
        static TEXEL: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
        images.push(SampledImageResource {
            binding,
            array_element: 0,
            descriptor_count: 1,
            width: 1,
            height: 1,
            layers: 1,
            kind: reims_vgpu_core::texture_shape::TextureKind::D2,
            multisampled: false,
            source: SampledSource::Bytes(Arc::clone(TEXEL.get_or_init(|| Arc::new(vec![0; 4])))),
            byte_origin: SampledByteOrigin::Synthetic,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
        crate::runtime::drain::note_store_route("synthesized_null_texture_served");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::meta::TextureShape;
    use metal2vulkan::reflect::{
        DescriptorLayout, DescriptorLocation, ResourceBinding, ShaderStage, REFLECTION_VERSION,
    };

    fn reflection() -> ShaderReflection {
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage: ShaderStage::Vertex,
            descriptor_layout: DescriptorLayout::default(),
            datalayout: None,
            bindings: vec![ResourceBinding {
                kind: ResourceKind::SynthesizedNullTexture,
                // Deliberately not the descriptor's offset in the texture band.
                metal_index: 77,
                descriptor: Some(DescriptorLocation {
                    set: 0,
                    binding: 37,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape: Some(TextureShape {
                    dimension: TextureDimension::D2,
                    arrayed: false,
                    multisampled: false,
                    component: TextureComponent::Float,
                    writable: false,
                    array_ref: false,
                    array_length: None,
                    storage_format: None,
                }),
                embedded_source: None,
                access: Some(ResourceAccess::Sampled),
                static_sampler: None,
            }],
            entry_point: None,
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: None,
            max_work_group_size: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            kernel_dispatch: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }
    }

    #[test]
    fn a_null_image_uses_the_reflected_binding_in_each_stage() {
        for (relocated, expected) in [
            (false, 37),
            (true, 37 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET),
        ] {
            let mut images = Vec::new();
            append_null_textures(&reflection(), relocated, &mut images).unwrap();
            assert_eq!(images.len(), 1);
            let image = &images[0];
            assert_eq!(image.binding, expected);
            assert_eq!((image.width, image.height, image.layers), (1, 1, 1));
            assert_eq!(image.kind, reims_vgpu_core::texture_shape::TextureKind::D2);
            assert_eq!(image.format, ash::vk::Format::R8G8B8A8_UNORM);
            assert_eq!(image.byte_origin, SampledByteOrigin::Synthetic);
            let SampledSource::Bytes(bytes) = &image.source else {
                panic!("synthetic image must own its bytes")
            };
            assert_eq!(bytes.as_slice(), &[0, 0, 0, 0]);
        }
    }

    #[test]
    fn the_runtime_wrapper_preserves_synthetic_refusal_identity() {
        use crate::backend::vulkan::engine::{DrawError, DrawPreparationDecline};
        let reason = SynthesizedImageDecline::ShapeUnsupported { binding: 37 };
        let error = DrawError::DrawPreparation(DrawPreparationDecline::SynthesizedSampled {
            stage: "vertex",
            reason,
        });
        assert_eq!(error.owner(), reason.owner());
        assert_eq!(
            crate::observe::Emit::decline("synthetic_test", &error).render(),
            "synthetic_test reason=draw_synth_null_shape_unsupported stage=vertex binding=37"
        );
    }

    #[test]
    fn an_ordinary_guest_texture_is_not_synthesized() {
        let mut reflection = reflection();
        reflection.bindings[0].kind = ResourceKind::Texture;
        let mut images = Vec::new();
        append_null_textures(&reflection, false, &mut images).unwrap();
        assert!(images.is_empty());
    }

    #[test]
    fn a_null_descriptor_never_overwrites_an_existing_binding() {
        let mut images = Vec::new();
        append_null_textures(&reflection(), false, &mut images).unwrap();
        assert_eq!(
            append_null_textures(&reflection(), false, &mut images),
            Err(SynthesizedImageDecline::BindingOccupied { binding: 37 })
        );
        assert_eq!(images.len(), 1);
    }

    #[test]
    fn unsupported_shapes_are_refused_instead_of_forged_as_2d() {
        for mutate in [
            (|s: &mut TextureShape| s.arrayed = true) as fn(&mut TextureShape),
            |s| s.dimension = TextureDimension::D3,
            |s| s.multisampled = true,
            |s| s.writable = true,
            |s| s.array_ref = true,
            |s| s.array_length = Some(2),
        ] {
            let mut reflection = reflection();
            mutate(reflection.bindings[0].texture_shape.as_mut().unwrap());
            assert_eq!(
                append_null_textures(&reflection, false, &mut Vec::new()),
                Err(SynthesizedImageDecline::ShapeUnsupported { binding: 37 })
            );
        }
    }

    #[test]
    fn malformed_synthetic_declarations_have_distinct_refusals() {
        use SynthesizedImageDecline as R;
        let cases: &[(fn(&mut ResourceBinding), R)] = &[
            (
                |r| r.descriptor = None,
                R::DescriptorMissing {
                    synthetic_index: 77,
                },
            ),
            (
                |r| r.descriptor.as_mut().unwrap().set = 1,
                R::DescriptorSet {
                    binding: 37,
                    set: 1,
                },
            ),
            (
                |r| r.descriptor.as_mut().unwrap().count = 2,
                R::DescriptorCount {
                    binding: 37,
                    count: 2,
                },
            ),
            (
                |r| r.descriptor.as_mut().unwrap().binding = SAMPLER_BINDING_BASE,
                R::BindingOutsideTextureBand {
                    binding: SAMPLER_BINDING_BASE,
                },
            ),
            (
                |r| r.param_index = Some(0),
                R::GuestParameter { binding: 37 },
            ),
            (|r| r.access = None, R::Access { binding: 37 }),
            (|r| r.texture_shape = None, R::ShapeMissing { binding: 37 }),
        ];
        let mut slugs = std::collections::BTreeSet::new();
        for (mutate, expected) in cases {
            let mut reflection = reflection();
            mutate(&mut reflection.bindings[0]);
            let refusal = append_null_textures(&reflection, false, &mut Vec::new()).unwrap_err();
            assert_eq!(refusal, *expected);
            assert!(slugs.insert(refusal.slug()));
        }
    }
}

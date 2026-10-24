//! Guest-visible publication for the MRT attachments the engine rendered.
//!
//! Readiness in the engine registry alone is not a Store: normal texture
//! consumers also ask the resource's writeback ledger which copy is current.
//! Retain each submitted identity and its pre-submit page authorization together.

use super::*;
use crate::backend::vulkan::engine::{DrawExecutionDecline, SecondaryColorTarget, TargetIdentity};

pub(super) fn load_seed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    color: &ColorRtRequest,
    identity: &TargetIdentity,
    format: ash::vk::Format,
) -> Option<crate::backend::vulkan::engine::GuestTargetSeed> {
    use crate::protocol::pass_action::LoadAction;
    let task_id = req.task_id;
    if LoadAction::from_declared(color.load_action) != LoadAction::Load {
        return None;
    }
    // LOAD at the encoder boundary asks whether guest memory or the resident
    // is current. Between draws of that encoder, the attachment is already
    // the resident. A DontCare Store has deliberately published no guest copy:
    // seeding from those bytes here erases the coverage the last draw produced.
    // Returning no seed selects SecondaryLoad::Resident; the engine still
    // refuses it if the resident has no initialized contents.
    if req.continues_render_pass {
        return None;
    }
    if color.target_gva != 0 {
        let span = GvaSpan {
            texture_ref: color.texture_ref,
            gva: color.target_gva,
            row_stride: color.row_stride,
            width: color.width,
            height: color.height,
            format: color.format,
        };
        if gva_resident_if_current(state, host, task_id, span)
            .is_ok_and(|current| current == *identity)
        {
            return None;
        }
        try_gva_target_guest_seed(state, host, task_id, color, format)
    } else {
        match resolve_mapper_ref_texture_load_seed(
            state,
            host,
            color.mapping_id,
            color.width,
            color.height,
            format,
        ) {
            Some(MapperRefTextureLoadSeed::Resident) => None,
            Some(MapperRefTextureLoadSeed::Guest(seed)) => Some(seed),
            Some(MapperRefTextureLoadSeed::Host(_, _)) | None => {
                try_mapper_ref_texture_target_guest_seed(
                    state,
                    host,
                    color.mapping_id,
                    color.width,
                    color.height,
                    format,
                )
            }
        }
    }
}

pub(super) struct SecondaryStore {
    target: ColorRtRequest,
    identity: TargetIdentity,
    allowed_pages: Option<super::super::StoreTargetPages>,
}

pub(super) fn prepare<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    colors: &[ColorRtRequest],
    targets: &[SecondaryColorTarget],
    task_id: u32,
) -> Vec<SecondaryStore> {
    targets
        .iter()
        .zip(colors.iter().skip(1))
        // A discarded scratch attachment has no publication obligation. In
        // particular its absent page authorization must not abort the primary
        // Store after the GPU has already rendered it. Keep the slot/identity
        // pair intact before filtering, as the pass still renders every slot.
        .filter(|(_, color)| {
            crate::protocol::pass_action::store_action_publishes_single_sample(color.store_action)
        })
        .map(|(secondary, color)| SecondaryStore {
            target: color.clone(),
            identity: secondary.identity.clone(),
            allowed_pages: sync_store_target_pages(state, host, task_id, color),
        })
        .collect()
}

pub(super) fn publish<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    stores: &[SecondaryStore],
) -> Result<(), DrawError> {
    for store in stores {
        let color = &store.target;
        let published = if color.target_gva != 0 {
            let armed = crate::backend::vulkan::gva_window(&store.identity).is_some_and(|window| {
                crate::runtime::writeback_debt::arm_gva(
                    crate::backend::vulkan::VulkanBackend,
                    state,
                    host,
                    task_id,
                    color,
                    window,
                )
            });
            if armed {
                true
            } else {
                match crate::runtime::render_writeback::vulkan::store_gva_frame(
                    state,
                    host,
                    task_id,
                    &store.identity,
                    color,
                    color.texture_ref,
                    store.allowed_pages.as_ref(),
                    &[],
                ) {
                    Ok(_) => true,
                    Err(decline) => {
                        crate::observe::Emit::decline("secondary_store", &decline).fail();
                        false
                    }
                }
            }
        } else {
            store_surface_resident(
                state,
                host,
                &store.identity,
                color.mapping_id,
                color.width,
                color.height,
                GuestStoreStatus {
                    guest_backed: false,
                    recorded: false,
                    footprint: None,
                },
            )
        };
        if !published {
            return Err(DrawError::DrawExecution(
                DrawExecutionDecline::SecondaryStoreNotPublished {
                    identity: store.identity.clone(),
                },
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::SecondaryLoad;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::protocol::pass_action::{MTL_STORE_ACTION_DONT_CARE, MTL_STORE_ACTION_STORE};
    use crate::runtime::host::FakeHost;

    #[test]
    fn continuing_a_discarded_mask_does_not_reload_its_unstored_guest_bytes() {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::protocol::pass_action::MTL_LOAD_ACTION_LOAD;
        use crate::protocol::pixel_format::{MTL_FORMAT_RG16_FLOAT, MTL_FORMAT_RGBA16_FLOAT};
        use crate::runtime::host::HostMemory;

        for format in [MTL_FORMAT_RG16_FLOAT, MTL_FORMAT_RGBA16_FLOAT] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.stable_map_pages = true;
            let page = 1_u64 << PAGE_SHIFT_X86;
            for pfn in [2, 3, 4] {
                host.map_range(pfn * page, page as usize, 0);
            }
            let mut dir = [0_u8; 8];
            st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], 3);
            st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
            host.write_gpa(2 * page, &dir).unwrap();
            host.put_u32(3 * page + 4, 4);
            state.define_task(1, 0x1_0000, 2);
            let color = ColorRtRequest {
                slot: 1,
                texture_ref: 11,
                target_gva: 0x1000,
                width: 8,
                height: 8,
                row_stride: 64,
                format,
                load_action: MTL_LOAD_ACTION_LOAD,
                store_action: MTL_STORE_ACTION_DONT_CARE,
                ..Default::default()
            };
            let (attachment, _) = translate::pixel::color_attachment(format).unwrap();
            let identity = secondary_gva_identity(&color, attachment.vk, 19);
            let mut request = DrawEncodeRequest {
                task_id: 1,
                ..Default::default()
            };
            assert!(load_seed(&mut state, &mut host, &request, &color, &identity, attachment.vk).is_some(),
                "a new encoder must still seed a LOAD from guest memory when no current resident exists");
            request.continues_render_pass = true;
            assert!(
                load_seed(
                    &mut state,
                    &mut host,
                    &request,
                    &color,
                    &identity,
                    attachment.vk
                )
                .is_none(),
                "an in-encoder scratch mask has not been stored to those guest bytes"
            );
            request.continues_render_pass = false;
            assert!(
                load_seed(
                    &mut state,
                    &mut host,
                    &request,
                    &color,
                    &identity,
                    attachment.vk
                )
                .is_some(),
                "the next encoder must recheck guest currency instead of trusting the old resident"
            );
        }
    }

    #[test]
    fn discarded_secondary_slots_do_not_require_publication_or_shift_stored_slots() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        let colors: Vec<_> = (0..4)
            .map(|slot| ColorRtRequest {
                slot,
                texture_ref: slot + 10,
                target_gva: 0x1000 * u64::from(slot + 1),
                row_stride: 32,
                width: 8,
                height: 8,
                format: crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                store_action: if slot == 0 || slot == 2 {
                    MTL_STORE_ACTION_STORE
                } else {
                    MTL_STORE_ACTION_DONT_CARE
                },
                ..Default::default()
            })
            .collect();
        let (attachment, _) = translate::pixel::color_attachment(colors[0].format).unwrap();
        let targets: Vec<_> = (1..4)
            .map(|slot| SecondaryColorTarget {
                identity: TargetIdentity::Anonymous { slot },
                width: 8,
                height: 8,
                attachment: attachment.with_clear([0.0; 4]),
                contents: SecondaryLoad::Clear,
                blend: None,
                color_write_mask: Default::default(),
            })
            .collect();
        let stores = prepare(&state, &host, &colors, &targets, 1);
        assert_eq!(stores.len(), 1, "discarded slots have no Store obligation");
        assert_eq!(stores[0].target.slot, 2);
        assert_eq!(stores[0].target.texture_ref, 12);
        assert_eq!(stores[0].identity, targets[1].identity);
    }
}

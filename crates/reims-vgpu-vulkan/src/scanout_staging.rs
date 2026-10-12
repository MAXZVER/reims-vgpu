//! Select a legal source for the CPU scanout's filtered presentation blit.
//!
//! Transfer support does not imply blit support. A host-written LINEAR image
//! may be copyable but not sampleable by a blit. On that arm, copy its bytes
//! into an OPTIMAL image before scaling; the formats and extent stay identical.

use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuBlitSource {
    Linear,
    CopyToOptimal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported {
    pub linear: vk::FormatFeatureFlags,
    pub optimal: vk::FormatFeatureFlags,
}

/// The CPU source uses LINEAR tiling and TRANSFER_SRC; the final blit uses
/// LINEAR filtering. The bridge additionally needs TRANSFER_DST.
pub fn cpu_blit_source(properties: vk::FormatProperties) -> Result<CpuBlitSource, Unsupported> {
    let transfer = vk::FormatFeatureFlags::TRANSFER_SRC;
    let blit =
        vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
    let linear = properties.linear_tiling_features;
    let optimal = properties.optimal_tiling_features;
    if linear.contains(transfer | blit) {
        return Ok(CpuBlitSource::Linear);
    }
    if linear.contains(transfer)
        && optimal.contains(transfer | vk::FormatFeatureFlags::TRANSFER_DST | blit)
    {
        return Ok(CpuBlitSource::CopyToOptimal);
    }
    Err(Unsupported { linear, optimal })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> vk::FormatFeatureFlags {
        vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
    }

    #[test]
    fn transfer_only_linear_storage_is_copied_before_filtering() {
        // The measured WSL Dozen BGRA8 answer. Its LINEAR image cannot be
        // passed to vkCmdBlitImage, even though vkCmdCopyImage succeeds.
        let properties = vk::FormatProperties {
            linear_tiling_features: vk::FormatFeatureFlags::from_raw(0xc000),
            optimal_tiling_features: vk::FormatFeatureFlags::from_raw(0xdd83),
            ..Default::default()
        };
        assert_eq!(
            cpu_blit_source(properties),
            Ok(CpuBlitSource::CopyToOptimal)
        );
    }

    #[test]
    fn a_linear_image_that_can_be_filtered_keeps_the_direct_path() {
        assert_eq!(
            cpu_blit_source(vk::FormatProperties {
                linear_tiling_features: features(),
                ..Default::default()
            }),
            Ok(CpuBlitSource::Linear)
        );
    }

    #[test]
    fn missing_linear_filter_support_also_requires_the_bridge() {
        assert_eq!(
            cpu_blit_source(vk::FormatProperties {
                linear_tiling_features: vk::FormatFeatureFlags::TRANSFER_SRC
                    | vk::FormatFeatureFlags::BLIT_SRC,
                optimal_tiling_features: features(),
                ..Default::default()
            }),
            Ok(CpuBlitSource::CopyToOptimal)
        );
    }

    #[test]
    fn incomplete_copy_or_filter_capabilities_are_refused() {
        let complete = features();
        for missing in [
            vk::FormatFeatureFlags::TRANSFER_DST,
            vk::FormatFeatureFlags::TRANSFER_SRC,
            vk::FormatFeatureFlags::BLIT_SRC,
            vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        ] {
            let properties = vk::FormatProperties {
                linear_tiling_features: vk::FormatFeatureFlags::TRANSFER_SRC,
                optimal_tiling_features: complete & !missing,
                ..Default::default()
            };
            assert_eq!(
                cpu_blit_source(properties),
                Err(Unsupported {
                    linear: properties.linear_tiling_features,
                    optimal: properties.optimal_tiling_features
                })
            );
        }
        assert!(cpu_blit_source(vk::FormatProperties {
            optimal_tiling_features: complete,
            ..Default::default()
        })
        .is_err());
    }
}

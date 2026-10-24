//! Native guest texels used to initialize a preserving MRT attachment.
use super::*;

pub(super) unsafe fn prepare(
    ctx: &super::super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
    gathers: &mut Vec<PendingGuestGather>,
) -> Result<Vec<Option<GuestTexels>>, DrawError> {
    req.secondary_targets
        .iter()
        .map(|secondary| {
            let Some(seed) = secondary.contents.guest_seed() else {
                return Ok(None);
            };
            if let Some(source) =
                import_sampled_guest_window(ctx, pools, counters, &seed.source, gathers)?
            {
                pools.note_guest_read_recorded();
                return Ok(Some(source));
            }
            let slot = pools.acquire_staging(ctx, seed.source.total_len, counters)?;
            pools.write_staging_from_runs(
                ctx,
                &slot,
                &seed.source.runs,
                seed.source.source_offset,
                seed.source.total_len,
            )?;
            counters.note_seed_upload(seed.source.total_len);
            Ok(Some(GuestTexels::Scratch(slot)))
        })
        .collect()
}

pub(super) struct Destination<'a> {
    pub attachment: &'a super::super::types::SecondaryColorTarget,
    pub image: vk::Image,
    pub access: super::super::pools::ResidentAccess,
    pub layout: vk::ImageLayout,
    pub feedback: bool,
}

pub(super) unsafe fn record(
    ctx: &super::super::context::DeviceContext,
    cb: vk::CommandBuffer,
    destination: Destination<'_>,
    source: &GuestTexels,
) {
    let seed = destination
        .attachment
        .contents
        .guest_seed()
        .expect("prepared guest seed");
    let (src_stage, src_access) = destination.access.source_scope();
    let before = [vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(destination.access.layout())
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(destination.image)
        .subresource_range(super::super::color_subresource_range())];
    super::super::cmd_sink::pipeline_barrier(
        &ctx.device,
        cb,
        src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &before,
    );
    let copies = [vk::BufferImageCopy::default()
        .buffer_offset(source.offset())
        .buffer_row_length(seed.source.row_length_texels)
        .image_subresource(super::super::color_subresource_layers())
        .image_extent(vk::Extent3D {
            width: destination.attachment.width,
            height: destination.attachment.height,
            depth: 1,
        })];
    super::super::cmd_sink::copy_buffer_to_image(
        &ctx.device,
        cb,
        source.buffer(),
        destination.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &copies,
    );
    let mut dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let mut dst_access =
        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    if destination.feedback {
        dst_stage |=
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
        dst_access |= vk::AccessFlags::SHADER_READ;
    }
    let after = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(dst_access)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(destination.layout)
        .image(destination.image)
        .subresource_range(super::super::color_subresource_range())];
    let dependency = if destination.feedback {
        vk::DependencyFlags::FEEDBACK_LOOP_EXT
    } else {
        vk::DependencyFlags::empty()
    };
    super::super::cmd_sink::pipeline_barrier(
        &ctx.device,
        cb,
        vk::PipelineStageFlags::TRANSFER,
        dst_stage,
        dependency,
        &[],
        &[],
        &after,
    );
}

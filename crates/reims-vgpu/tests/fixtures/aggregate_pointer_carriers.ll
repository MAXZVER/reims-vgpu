target triple = "air64-apple-macosx15.0.0"
%Handle = type { ptr addrspace(1) }
%Sampler = type { ptr addrspace(2) }
%Bundle = type { %Handle, %Sampler, ptr addrspace(2), <2 x float> }
%Matrix = type { [4 x <4 x float>] }

define internal fastcc %Bundle @make_bundle(ptr addrspace(1) %image, ptr addrspace(2) %sampler, ptr addrspace(2) %config, <2 x float> %coord) {
entry:
  %a = insertvalue %Bundle poison, ptr addrspace(1) %image, 0, 0
  %b = insertvalue %Bundle %a, ptr addrspace(2) %sampler, 1, 0
  %c = insertvalue %Bundle %b, ptr addrspace(2) %config, 2
  %d = insertvalue %Bundle %c, <2 x float> %coord, 3
  ret %Bundle %d
}

define void @k(ptr addrspace(1) %image, ptr addrspace(2) %config, ptr addrspace(1) %out, ptr addrspace(2) %sampler) {
entry:
  %base = load <4 x float>, ptr addrspace(2) %config, align 16
  %offset = getelementptr i8, ptr addrspace(2) %config, i64 16
  %bundle = call fastcc %Bundle @make_bundle(ptr addrspace(1) %image, ptr addrspace(2) %sampler, ptr addrspace(2) %offset, <2 x float> <float 0.375, float 0.625>)
  %wrapper = extractvalue %Bundle %bundle, 0
  %src = extractvalue %Handle %wrapper, 0
  %swrapper = extractvalue %Bundle %bundle, 1
  %ptr = extractvalue %Bundle %bundle, 2
  %coord = extractvalue %Bundle %bundle, 3
  %columnptr = getelementptr %Matrix, ptr addrspace(2) %ptr, i64 0, i32 0, i64 1
  %column = load <4 x float>, ptr addrspace(2) %columnptr, align 16
  %result = call fastcc <4 x float> @consume(%Handle %wrapper, %Sampler %swrapper, <4 x float> %column, <2 x float> %coord)
  %sum = fadd <4 x float> %result, %base
  store <4 x float> %sum, ptr addrspace(1) %out, align 16
  ret void
}

define internal fastcc <4 x float> @consume(%Handle %image, %Sampler %sampler, <4 x float> %column, <2 x float> %coord) {
entry:
  %src = extractvalue %Handle %image, 0
  %s = extractvalue %Sampler %sampler, 0
  %sample = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) %s, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  %result = fadd <4 x float> %color, %column
  ret <4 x float> %result
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"image"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"void", !"air.arg_name", !"config"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}

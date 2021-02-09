function results = compute_bsseval_v2(srcs, refs)
    results = zeros(size(srcs,1),3);
    for ii=1:size(srcs,1)
        [e1,e2,e3] = bss_decomp_gain(srcs(ii,:), ii, refs);
        [src_sdr,src_sir,src_sar] = bss_crit(e1,e2,e3);
        results(ii,1) = src_sdr;
        results(ii,2) = src_sir;
        results(ii,3) = src_sar;
    end
end
# models/network.py
"""网络结构构建模块: 连接矩阵 + D1/D2 受体 Mask 分配"""
import torch
import config


def create_network_structure(N_E: int, N_I: int, device: torch.device):
    """
    构建 E-I 随机稀疏连接网络并分配 D1/D2 受体 Mask。
    
    参数来自 config 模块 (连接概率、权重、受体比例)，
    但 N_E / N_I / device 由调用方传入以保持灵活性。

    Returns:
        W_t       : (N, N) 转置后的连接矩阵，供 (Batch, N) @ (N, N) 使用
        mask_d1   : (N,)   bool, D1 受体 Mask
        mask_d2   : (N,)   bool, D2 受体 Mask
        groups_info: dict   亚群边界索引
    """
    N = N_E + N_I

    # ---- 1. 连接矩阵 ----
    W = torch.zeros((N, N), device=device)
    mask_conn = (torch.rand((N, N), device=device) < config.CONN_PROB).float()
    mask_conn.fill_diagonal_(0)

    W[:, :N_E] = config.W_EXC * mask_conn[:, :N_E]
    W[:, N_E:] = config.W_INH * mask_conn[:, N_E:]

    # 转置: 后续用 spikes @ W_t -> (Batch, N)
    W_t = W.t().contiguous()

    # ---- 2. D1 / D2 受体 Mask ----
    mask_d1 = torch.zeros(N, dtype=torch.bool, device=device)
    mask_d2 = torch.zeros(N, dtype=torch.bool, device=device)

    # 兴奋性神经元
    n_e_d1 = int(N_E * config.FRAC_E_D1)
    n_e_d2 = int(N_E * config.FRAC_E_D2)
    idx_e_d1_end = n_e_d1
    idx_e_d2_end = n_e_d1 + n_e_d2
    mask_d1[:idx_e_d1_end] = True
    mask_d2[idx_e_d1_end:idx_e_d2_end] = True

    # 抑制性神经元
    n_i_d1 = int(N_I * config.FRAC_I_D1)
    n_i_d2 = int(N_I * config.FRAC_I_D2)
    idx_i_start = N_E
    idx_i_d1_end = idx_i_start + n_i_d1
    idx_i_d2_end = idx_i_d1_end + n_i_d2
    mask_d1[idx_i_start:idx_i_d1_end] = True
    mask_d2[idx_i_d1_end:idx_i_d2_end] = True

    groups_info = {
        'e_d1_end': idx_e_d1_end,
        'e_d2_end': idx_e_d2_end,
        'e_other_end': N_E,
    }
    return W_t, mask_d1, mask_d2, groups_info


# ==============================================================================
#  Working-Memory network: structured selective sub-pools (Scheme A)
#
#  Layout (indices, with default config: N_E=824, N_I=176, pool_size=60):
#      [0,            173)  E-D1     (DA D1)        ── carved into:
#         [0,         60)         Mem-A   (cue receiver, P = WM_POOL_SIZE)
#         [60,       120)         Mem-B   (competing pool)
#         [120,      173)         D1-BG   (D1 background, no special role)
#      [173,         379)  E-D2     (DA D2)
#      [379,         824)  E-Other  (no DA receptor)
#      [824,         876)  I-D1     (first 30% of N_I, FRAC_I_D1=0.30)
#      [876,         890)  I-D2     (next 8%  of N_I, FRAC_I_D2=0.08)
#      [890,         940)  I-Other  (no DA receptor, shared global inhibition)
#      [940,        1000)  I-WM     (LAST iwm_size cells of N_I; carved from
#                                    the no-receptor I-Other tail so the WTA
#                                    feedback pool is *not* DA-modulated)
#
#  ⚠️  Critical placement note (post-mortem of "DA Δ ≈ 1 Hz" runs):
#      I-WM used to live at the FRONT of N_I [824, 824+iwm_size), which
#      overlapped with I-D1 [824, 876).  That meant DA increases
#      simultaneously boosted Mem-A excitation AND I-WM-mediated feedback
#      inhibition, the two cancelled each other and DA modulation barely
#      moved Mem-A.  Putting I-WM at the TAIL puts it inside I-Other (no
#      D1/D2 receptors), so DA only acts on the Mem→Mem path and the
#      WTA loop is DA-neutral.
#
#  Rationale for the pool placement:
#    - Putting Mem-A/B in E-D1 makes the two competing memory pools
#      *receptor-homogeneous*: both feel D1 modulation symmetrically.
#      DA increases therefore drive both attractors equally, and the
#      winner is determined by the cue (input selectivity) rather than
#      by an artificial receptor asymmetry.  This matches the standard
#      Wang/Brunel WM model and the Vijayraghavan 2007 inverted-U story
#      (D1 stimulation enhances delay firing up to a point, then
#      degrades selectivity when both pools breach the WTA threshold).
#    - The base sparse connectivity is preserved (so the checkpoint
#      fingerprint remains valid).  We *additionally* densify the
#      intra-pool blocks Mem-A↔Mem-A and Mem-B↔Mem-B to support
#      persistent recurrent activity.
# ==============================================================================
def create_wm_network(N_E: int, N_I: int, device: torch.device,
                      pool_size: int = None,
                      intra_prob: float = None,
                      intra_w: float = None,
                      cross_prob: float = None,
                      use_wta: bool = None,
                      iwm_size: int = None,
                      e2i_prob: float = None,
                      e2i_w: float = None,
                      i2e_prob: float = None,
                      i2e_w: float = None):
    """
    Build a WM-augmented network.

    Returns
    -------
    W_t          : (N, N) transposed weight matrix (same convention as base).
    mask_d1      : (N,) bool, D1 receptor mask (identical to base network).
    mask_d2      : (N,) bool, D2 receptor mask (identical to base network).
    groups_info  : dict, sub-population boundaries; *additionally* contains
                   the WM pool boundaries: 'mem_a_start', 'mem_a_end',
                   'mem_b_start', 'mem_b_end', 'pool_size', and (when WTA
                   is enabled) 'iwm_start', 'iwm_end'.

    WTA structure
    -------------
    When `use_wta=True` (default), an "I-WM" sub-pool is carved from the
    front of the inhibitory population N_I, and the shared inhibition
    loop Mem-{A,B} <-> I-WM is densified.  This is what enforces the
    winner-take-all competition between Mem-A and Mem-B: any pool that
    becomes hyperactive recruits I-WM, which in turn presses both pools
    back down — only a pool with strong intrinsic recurrence (the cued
    one) survives.
    """
    if pool_size is None:
        pool_size = config.WM_POOL_SIZE
    if intra_prob is None:
        intra_prob = config.WM_INTRA_PROB
    if intra_w is None:
        intra_w = config.WM_INTRA_W
    if cross_prob is None:
        cross_prob = config.WM_CROSS_PROB
    if use_wta is None:
        use_wta = config.WM_USE_WTA
    if iwm_size is None:
        iwm_size = config.WM_IWM_SIZE
    if e2i_prob is None:
        e2i_prob = config.WM_E2I_PROB
    if e2i_w is None:
        e2i_w = config.WM_E2I_W
    if i2e_prob is None:
        i2e_prob = config.WM_I2E_PROB
    if i2e_w is None:
        i2e_w = config.WM_I2E_W

    # 1. Start from the base network (so the unmodified part stays identical
    #    to a regular run with the same seed).
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(N_E, N_I, device)

    # 2. Compute pool boundaries within E-D1 segment.
    #    Mem-A/B both live in [0, e_d1_end) so they share D1R receptor
    #    modulation (receptor-homogeneous competition; see header comment).
    e_d1_start = 0
    e_d1_end   = groups_info['e_d1_end']           # = idx_e_d1_end
    e_d1_size  = e_d1_end - e_d1_start
    if 2 * pool_size > e_d1_size:
        raise ValueError(
            f"E-D1 segment has only {e_d1_size} neurons but 2*pool_size="
            f"{2*pool_size} requested. Reduce WM_POOL_SIZE or increase FRAC_E_D1."
        )

    mem_a_start = e_d1_start
    mem_a_end   = mem_a_start + pool_size
    mem_b_start = mem_a_end
    mem_b_end   = mem_b_start + pool_size

    # 3. Re-densify the intra-pool blocks.
    #    W is laid out as W[post, pre].  We stored W_t = W.T -> W_t[pre, post].
    #    We modify W in (post, pre) convention then re-transpose back.
    W = W_t.t().contiguous().clone()

    def _densify_block(W_local: torch.Tensor, post_lo: int, post_hi: int,
                       pre_lo: int, pre_hi: int, prob: float, weight: float):
        """Overwrite W[post_lo:post_hi, pre_lo:pre_hi] with a freshly drawn
        sparse block of the given probability/weight; diagonal kept zero."""
        size_post = post_hi - post_lo
        size_pre  = pre_hi  - pre_lo
        if size_post <= 0 or size_pre <= 0:
            return
        block = (torch.rand((size_post, size_pre), device=W_local.device) < prob).float() * weight
        # Zero the diagonal contribution if the block is on the main diagonal
        if post_lo == pre_lo and post_hi == pre_hi:
            n = size_post
            block[torch.arange(n, device=W_local.device),
                  torch.arange(n, device=W_local.device)] = 0.0
        W_local[post_lo:post_hi, pre_lo:pre_hi] = block

    # Strong recurrence within each memory pool
    _densify_block(W, mem_a_start, mem_a_end, mem_a_start, mem_a_end,
                   intra_prob, intra_w)
    _densify_block(W, mem_b_start, mem_b_end, mem_b_start, mem_b_end,
                   intra_prob, intra_w)
    # Optional cross-pool connectivity (default 0 -> hard competition)
    if cross_prob > 0.0:
        _densify_block(W, mem_a_start, mem_a_end, mem_b_start, mem_b_end,
                       cross_prob, intra_w * 0.5)
        _densify_block(W, mem_b_start, mem_b_end, mem_a_start, mem_a_end,
                       cross_prob, intra_w * 0.5)

    # ── 3b. WTA: shared-inhibition loop Mem-{A,B} <-> I-WM ──
    #
    # Layout reminder: indices [N_E, N_E + N_I) are the inhibitory population.
    # We claim the *LAST* `iwm_size` of them as the dedicated I-WM sub-pool,
    # so the I-WM block lands inside I-Other (no D1/D2 receptors).  This is
    # critical: a DA-neutral feedback pool means the inverted-U story is
    # mediated purely through E-D1 → Mem excitation, not contaminated by
    # DA-driven feedback inhibition (see header comment for post-mortem).
    #
    # Two directional blocks are densified:
    #
    #     E2I : (post=I-WM, pre=Mem-A ∪ Mem-B)        — excitatory drive
    #     I2E : (post=Mem-A ∪ Mem-B, pre=I-WM)        — inhibitory feedback
    #
    # Dale's principle is preserved: I-WM cells are inhibitory, so the
    # I2E weight is *negative*; Mem-{A,B} cells are excitatory, so the
    # E2I weight is *positive*.
    N_total   = N_E + N_I
    iwm_end   = N_total                              # tail of N_I
    iwm_start = N_total - iwm_size if use_wta else N_total
    if use_wta:
        if iwm_size <= 0 or iwm_size > N_I:
            raise ValueError(
                f"WM_IWM_SIZE={iwm_size} must be in (0, N_I={N_I}].")
        # Sanity-check: I-WM must not collide with I-D1 / I-D2 receptor masks.
        n_i_d1 = int(N_I * config.FRAC_I_D1)
        n_i_d2 = int(N_I * config.FRAC_I_D2)
        i_receptor_end = N_E + n_i_d1 + n_i_d2          # end of I-D1∪I-D2
        if iwm_start < i_receptor_end:
            raise ValueError(
                f"I-WM tail-placement [{iwm_start},{iwm_end}) overlaps with "
                f"I-D1∪I-D2 [{N_E},{i_receptor_end}). Reduce WM_IWM_SIZE "
                f"(currently {iwm_size}) so it fits inside I-Other "
                f"[{i_receptor_end},{N_total}) (max {N_total - i_receptor_end}).")

        # E -> I (excitatory), separately for Mem-A and Mem-B for clarity
        _densify_block(W, iwm_start, iwm_end, mem_a_start, mem_a_end,
                       e2i_prob, e2i_w)
        _densify_block(W, iwm_start, iwm_end, mem_b_start, mem_b_end,
                       e2i_prob, e2i_w)
        # I -> E (inhibitory feedback)
        _densify_block(W, mem_a_start, mem_a_end, iwm_start, iwm_end,
                       i2e_prob, i2e_w)
        _densify_block(W, mem_b_start, mem_b_end, iwm_start, iwm_end,
                       i2e_prob, i2e_w)

    W_t_new = W.t().contiguous()

    # 3c. Split W into NMDA (slow, only Mem-A↔Mem-A and Mem-B↔Mem-B intra
    #     blocks) and AMPA (everything else, including E-BG, I, E2I, I2E).
    #
    #     This is the dual-channel synapse design (Wang 2002):
    #       - NMDA-style slow integrator (~100 ms) only inside memory pools,
    #         gives the recurrent excitation enough memory to support
    #         persistent activity (attractor).
    #       - AMPA/GABA fast (~5 ms) elsewhere, so I-WM feedback can react
    #         to Mem activity within a few ms (essential for WTA selectivity).
    #
    #     Important: the split is done on the (post, pre) layout matrix `W`
    #     before transposing.
    W_nmda = torch.zeros_like(W)
    W_nmda[mem_a_start:mem_a_end, mem_a_start:mem_a_end] = \
        W[mem_a_start:mem_a_end, mem_a_start:mem_a_end]
    W_nmda[mem_b_start:mem_b_end, mem_b_start:mem_b_end] = \
        W[mem_b_start:mem_b_end, mem_b_start:mem_b_end]
    W_ampa = W - W_nmda

    W_ampa_t = W_ampa.t().contiguous()
    W_nmda_t = W_nmda.t().contiguous()

    # 4. Augment groups_info with pool indices.
    groups_info = dict(groups_info)
    groups_info.update({
        'mem_a_start': mem_a_start,
        'mem_a_end':   mem_a_end,
        'mem_b_start': mem_b_start,
        'mem_b_end':   mem_b_end,
        'pool_size':   pool_size,
        'wm_intra_prob': intra_prob,
        'wm_intra_w':    intra_w,
        'use_wta':       bool(use_wta),
        'iwm_start':     iwm_start,
        'iwm_end':       iwm_end,
        'iwm_size':      iwm_end - iwm_start,
        'wm_e2i_prob':   e2i_prob,
        'wm_e2i_w':      e2i_w,
        'wm_i2e_prob':   i2e_prob,
        'wm_i2e_w':      i2e_w,
        # Dual-channel split: NMDA on intra-pool E→E, AMPA elsewhere
        'W_ampa_t':      W_ampa_t,
        'W_nmda_t':      W_nmda_t,
    })
    return W_t_new, mask_d1, mask_d2, groups_info


def build_wm_stim_mask(groups_info: dict, target_pool: str, N_total: int,
                       device: torch.device) -> torch.Tensor:
    """
    Build a (N,) float tensor with 1.0 at the indices of the requested
    memory pool and 0.0 elsewhere.  Used as `stim_mask` for cue injection.

    target_pool : "A" or "B"
    """
    mask = torch.zeros(N_total, device=device)
    if target_pool.upper() == 'A':
        mask[groups_info['mem_a_start']:groups_info['mem_a_end']] = 1.0
    elif target_pool.upper() == 'B':
        mask[groups_info['mem_b_start']:groups_info['mem_b_end']] = 1.0
    else:
        raise ValueError(f"Unknown pool '{target_pool}', expected 'A' or 'B'.")
    return mask

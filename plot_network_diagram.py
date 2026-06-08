"""
DA-PFC Full Network Architecture Diagram (Wang 2002 style).

Draws all 9 sub-populations with their connectivity in a clean,
publication-quality style using nested circles and curved arrows.

Network layout (N=1000):
  Excitatory (N_E=824):
    E-D1 [0, 173):   Mem-A [0,60), Mem-B [60,120), D1-BG [120,173)
    E-D2 [173, 379):  D2 receptor
    E-Other [379, 824): no receptor
  Inhibitory (N_I=176):
    I-D1 [824, 876):  D1 receptor
    I-D2 [876, 890):  D2 receptor
    I-Other [890, 940): no receptor
    I-WM [940, 1000): WTA shared inhibition (DA-neutral)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
import numpy as np


def draw_full_network(save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-2, 16)
    ax.set_ylim(-3, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # ====================================================================
    # Color palette
    # ====================================================================
    C_D1 = '#C0392B'       # D1 receptor color (warm red)
    C_D2 = '#2980B9'       # D2 receptor color (cool blue)
    C_NONE = '#7F8C8D'     # No receptor (gray)
    C_MEMA = '#E74C3C'     # Mem-A (bright red)
    C_MEMB = '#3498DB'     # Mem-B (bright blue)
    C_IWM = '#2C3E50'      # I-WM (dark navy)
    C_DA = '#8E44AD'       # DA modulation (purple)
    C_EXC = '#27AE60'      # Excitatory arrow (green)
    C_INH = '#E67E22'      # Inhibitory arrow (orange)

    # ====================================================================
    # 1. Large enclosing ellipse: ALL Excitatory neurons (N_E=824)
    # ====================================================================
    exc_ellipse = mpatches.Ellipse((6.0, 4.5), 11.0, 8.5, fill=False,
                                    edgecolor='#2E8B57', linewidth=2.5,
                                    linestyle='-', zorder=1)
    ax.add_patch(exc_ellipse)
    ax.text(6.0, 0.0, 'Excitatory (N=824)', fontsize=10,
            color='#2E8B57', ha='center', va='center', style='italic')

    # ====================================================================
    # 2. E-D1 sub-region (enclosing circle for Mem-A, Mem-B, D1-BG)
    # ====================================================================
    ed1_circle = plt.Circle((3.5, 4.5), 3.2, fill=False,
                            edgecolor=C_D1, linewidth=1.8,
                            linestyle='--', alpha=0.6, zorder=2)
    ax.add_patch(ed1_circle)
    ax.text(3.5, 1.1, 'E-D1 (173)', fontsize=8, color=C_D1,
            ha='center', va='center', alpha=0.8)

    # --- Mem-A ---
    mem_a = plt.Circle((2.2, 5.2), 1.1, fill=False,
                       edgecolor=C_MEMA, linewidth=2.2, zorder=3)
    ax.add_patch(mem_a)
    ax.text(2.2, 5.2, 'Mem-A\n(60)', fontsize=11, fontweight='bold',
            color=C_MEMA, ha='center', va='center')

    # --- Mem-B ---
    mem_b = plt.Circle((4.8, 5.2), 1.1, fill=False,
                       edgecolor=C_MEMB, linewidth=2.2, zorder=3)
    ax.add_patch(mem_b)
    ax.text(4.8, 5.2, 'Mem-B\n(60)', fontsize=11, fontweight='bold',
            color=C_MEMB, ha='center', va='center')

    # --- D1-BG ---
    d1bg = plt.Circle((3.5, 3.0), 0.7, fill=False,
                      edgecolor=C_D1, linewidth=1.5, zorder=3)
    ax.add_patch(d1bg)
    ax.text(3.5, 3.0, 'D1-BG\n(53)', fontsize=8, color=C_D1,
            ha='center', va='center')

    # ====================================================================
    # 3. E-D2 sub-population
    # ====================================================================
    ed2 = plt.Circle((7.8, 3.5), 1.0, fill=False,
                     edgecolor=C_D2, linewidth=1.8, zorder=3)
    ax.add_patch(ed2)
    ax.text(7.8, 3.5, 'E-D2\n(206)', fontsize=9, fontweight='bold',
            color=C_D2, ha='center', va='center')

    # ====================================================================
    # 4. E-Other sub-population
    # ====================================================================
    eother = plt.Circle((9.5, 5.5), 1.2, fill=False,
                        edgecolor=C_NONE, linewidth=1.5, zorder=3)
    ax.add_patch(eother)
    ax.text(9.5, 5.5, 'E-Other\n(445)', fontsize=9, color=C_NONE,
            ha='center', va='center')

    # ====================================================================
    # 5. Inhibitory populations (right side, separate cluster)
    # ====================================================================
    # Enclosing region for all inhibitory
    inh_ellipse = mpatches.Ellipse((13.0, 5.0), 4.5, 7.0, fill=False,
                                    edgecolor=C_IWM, linewidth=2.0,
                                    linestyle='-', alpha=0.7, zorder=1)
    ax.add_patch(inh_ellipse)
    ax.text(13.0, 1.2, 'Inhibitory (N=176)', fontsize=10,
            color=C_IWM, ha='center', va='center', style='italic')

    # --- I-WM (the key WTA pool, DA-neutral) ---
    iwm = plt.Circle((13.0, 6.5), 1.2, fill=False,
                     edgecolor=C_IWM, linewidth=2.5, zorder=3)
    ax.add_patch(iwm)
    ax.text(13.0, 6.5, 'I-WM\n(60)', fontsize=11, fontweight='bold',
            color=C_IWM, ha='center', va='center')

    # --- I-D1 ---
    id1 = plt.Circle((12.0, 4.2), 0.7, fill=False,
                     edgecolor=C_D1, linewidth=1.5, zorder=3)
    ax.add_patch(id1)
    ax.text(12.0, 4.2, 'I-D1\n(52)', fontsize=8, color=C_D1,
            ha='center', va='center')

    # --- I-D2 ---
    id2 = plt.Circle((14.0, 4.2), 0.6, fill=False,
                     edgecolor=C_D2, linewidth=1.5, zorder=3)
    ax.add_patch(id2)
    ax.text(14.0, 4.2, 'I-D2\n(14)', fontsize=8, color=C_D2,
            ha='center', va='center')

    # --- I-Other ---
    iother = plt.Circle((13.0, 2.8), 0.7, fill=False,
                        edgecolor=C_NONE, linewidth=1.5, zorder=3)
    ax.add_patch(iother)
    ax.text(13.0, 2.8, 'I-Other\n(50)', fontsize=8, color=C_NONE,
            ha='center', va='center')

    # ====================================================================
    # 6. Self-recurrence arrows (NMDA, Wang 2002 attractor core)
    # ====================================================================
    # Mem-A self-loop
    arrow_a_self = FancyArrowPatch(
        (1.3, 6.1), (3.0, 6.1),
        arrowstyle='->', mutation_scale=15,
        connectionstyle='arc3,rad=-0.7',
        color=C_MEMA, lw=2.2, zorder=5
    )
    ax.add_patch(arrow_a_self)

    # Mem-B self-loop
    arrow_b_self = FancyArrowPatch(
        (3.9, 6.1), (5.6, 6.1),
        arrowstyle='->', mutation_scale=15,
        connectionstyle='arc3,rad=-0.7',
        color=C_MEMB, lw=2.2, zorder=5
    )
    ax.add_patch(arrow_b_self)

    # NMDA label
    ax.text(3.5, 7.2, 'NMDA\n(p=0.5, W=4.0)', fontsize=7,
            color='#555', ha='center', va='center')

    # ====================================================================
    # 7. WTA loop: Mem -> I-WM (excitatory) and I-WM -> Mem (inhibitory)
    # ====================================================================
    # Mem-A -> I-WM
    arrow_a2iwm = FancyArrowPatch(
        (3.3, 5.5), (11.8, 6.8),
        arrowstyle='->', mutation_scale=15,
        connectionstyle='arc3,rad=-0.15',
        color=C_EXC, lw=1.8, zorder=4
    )
    ax.add_patch(arrow_a2iwm)

    # Mem-B -> I-WM
    arrow_b2iwm = FancyArrowPatch(
        (5.9, 5.5), (11.8, 6.3),
        arrowstyle='->', mutation_scale=15,
        connectionstyle='arc3,rad=-0.1',
        color=C_EXC, lw=1.8, zorder=4
    )
    ax.add_patch(arrow_b2iwm)

    # I-WM -> Mem-A (inhibitory, dashed)
    arrow_iwm2a = FancyArrowPatch(
        (11.8, 6.0), (3.0, 4.5),
        arrowstyle='-|>', mutation_scale=12,
        connectionstyle='arc3,rad=-0.2',
        color=C_INH, lw=2.0, linestyle='--', zorder=4
    )
    ax.add_patch(arrow_iwm2a)

    # I-WM -> Mem-B (inhibitory, dashed)
    arrow_iwm2b = FancyArrowPatch(
        (11.8, 5.8), (5.5, 4.5),
        arrowstyle='-|>', mutation_scale=12,
        connectionstyle='arc3,rad=-0.15',
        color=C_INH, lw=2.0, linestyle='--', zorder=4
    )
    ax.add_patch(arrow_iwm2b)

    # WTA labels
    ax.text(8.0, 7.8, r'$W_{E \to I}$' + '\n(p=0.2, W=3.0)',
            fontsize=7, color=C_EXC, ha='center', va='center')
    ax.text(8.0, 3.2, r'$W_{I \to E}$' + '\n(p=0.15, W=-3.5)',
            fontsize=7, color=C_INH, ha='center', va='center')

    # ====================================================================
    # 8. Background sparse connectivity (light, between all populations)
    # ====================================================================
    # Thin gray bidirectional arrow between E and I clusters
    arrow_bg = FancyArrowPatch(
        (10.5, 5.0), (11.0, 5.0),
        arrowstyle='<->', mutation_scale=10,
        connectionstyle='arc3,rad=0',
        color='#BDC3C7', lw=1.2, zorder=2
    )
    ax.add_patch(arrow_bg)
    ax.text(10.7, 4.4, 'sparse\np=0.02', fontsize=6,
            color='#95A5A6', ha='center', va='center')

    # ====================================================================
    # 9. External inputs (from top)
    # ====================================================================
    # I_A -> Mem-A
    ax.annotate('', xy=(2.2, 6.3), xytext=(2.2, 9.5),
                arrowprops=dict(arrowstyle='->', color=C_MEMA, lw=2.5))
    ax.text(2.2, 9.8, r'$I_A$' + '\n(cue)', fontsize=12, fontweight='bold',
            color=C_MEMA, ha='center', va='bottom')

    # I_B -> Mem-B
    ax.annotate('', xy=(4.8, 6.3), xytext=(4.8, 9.5),
                arrowprops=dict(arrowstyle='->', color=C_MEMB, lw=2.5))
    ax.text(4.8, 9.8, r'$I_B$', fontsize=12, fontweight='bold',
            color=C_MEMB, ha='center', va='bottom')

    # Noise -> all
    ax.annotate('', xy=(7.0, 8.5), xytext=(7.0, 10.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(7.0, 10.8, 'Noise', fontsize=10, color='black',
            ha='center', va='bottom')

    # ====================================================================
    # 10. DA modulation (purple arrows to D1 and D2 populations)
    # ====================================================================
    # DA -> E-D1 region
    ax.annotate('', xy=(1.0, 3.5), xytext=(-1.5, 9.0),
                arrowprops=dict(arrowstyle='->', color=C_DA,
                                lw=2.5, linestyle='-'))
    ax.text(-1.5, 9.5, 'DA', fontsize=13, fontweight='bold',
            color=C_DA, ha='center', va='bottom')

    # D1R label
    ax.text(-0.8, 6.5, 'D1R', fontsize=9, color=C_DA,
            ha='center', va='center', rotation=55)

    # DA -> E-D2 (smaller arrow)
    ax.annotate('', xy=(7.2, 4.5), xytext=(5.5, 8.5),
                arrowprops=dict(arrowstyle='->', color=C_DA,
                                lw=1.5, linestyle='--', alpha=0.6))
    ax.text(5.8, 8.0, 'D2R', fontsize=8, color=C_DA,
            ha='center', va='center', alpha=0.7)

    # ====================================================================
    # 11. SFA indicator on Mem pools
    # ====================================================================
    ax.text(0.5, 4.0, 'SFA', fontsize=7, color='#888',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5',
                      edgecolor='#CCC', linewidth=0.5))

    # ====================================================================
    # 12. Legend
    # ====================================================================
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor=C_D1, linewidth=1.5,
                       label='D1 receptor'),
        mpatches.Patch(facecolor='white', edgecolor=C_D2, linewidth=1.5,
                       label='D2 receptor'),
        mpatches.Patch(facecolor='white', edgecolor=C_NONE, linewidth=1.5,
                       label='No receptor (DA-neutral)'),
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color=C_EXC,
                            label='Excitatory (AMPA)'),
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color=C_INH,
                            label='Inhibitory (GABA)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
              framealpha=0.9, edgecolor='#DDD')

    # ====================================================================
    # Title
    # ====================================================================
    ax.set_title("DA-PFC Working Memory Network Architecture\n"
                 "N=1000 (E=824, I=176) | Dual-channel: AMPA + NMDA",
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    draw_full_network('wang2002_style_network.png')

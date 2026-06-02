"""Generate MARST_architecture.drawio -- editable in draw.io / app.diagrams.net /
VS Code Draw.io extension. Mirrors the matplotlib v5 architecture diagram with
all the same nodes, arrows, colors, and labels."""
import html

# Colors (matching the matplotlib version)
COL_INPUT     = '#E8EAED'; COL_INPUT_BD  = '#5F6368'
COL_ANCHOR    = '#FDE0B2'; COL_ANCHOR_BD = '#B45F06'
COL_AUX       = '#F0E5C3'; COL_AUX_BD    = '#8C7000'
COL_FEAT      = '#F7E9C1'; COL_FEAT_BD   = '#7F6000'
COL_TRUNK     = '#BFD9EE'; COL_TRUNK_BD  = '#0F4A6B'
COL_PI        = '#D5E8D4'; COL_PI_BD     = '#2D6F36'
COL_R         = '#E1D5E7'; COL_R_BD      = '#6A2E84'
COL_ALPHA     = '#C8E6E1'; COL_ALPHA_BD  = '#0F6B5E'
COL_COMP      = '#FFEFD5'; COL_COMP_BD   = '#B8830B'
COL_OUTPUT    = '#FAD2D2'; COL_OUTPUT_BD = '#991F1F'
COL_BYPASS    = '#B45F06'
COL_RAW       = '#3C72A0'
COL_ARROW     = '#222222'

# Style builders
def box_style(fill, border, fs=12, bold=False, italic=False, tc='#111111'):
    fstyle = 0
    if bold:   fstyle |= 1
    if italic: fstyle |= 2
    return (f"rounded=1;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={border};"
            f"fontColor={tc};fontSize={fs};fontStyle={fstyle};"
            f"strokeWidth=1.5;arcSize=14;")

def text_style(fs=12, bold=False, italic=False, color='#111111', align='center'):
    fstyle = 0
    if bold:   fstyle |= 1
    if italic: fstyle |= 2
    return (f"text;html=1;strokeColor=none;fillColor=none;"
            f"fontColor={color};fontSize={fs};fontStyle={fstyle};"
            f"align={align};verticalAlign=middle;")

def edge_style(color=COL_ARROW, sw=2.0, curved=False, dashed=False):
    s = (f"endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=0.5;exitDx=0;exitDy=0;"
         f"entryX=0.5;entryY=0.5;entryDx=0;entryDy=0;strokeColor={color};strokeWidth={sw};"
         f"endSize=8;endFill=1;")
    if curved:
        s += "curved=1;"
    if dashed:
        s += "dashed=1;"
    return s

# Build cells
cells = []
next_id = [2]   # 0 and 1 are reserved by draw.io

def add_box(x, y, w, h, value, style):
    """Add a vertex (box) cell. Returns its id."""
    cid = next_id[0]; next_id[0] += 1
    val = html.escape(value).replace('\n', '<br>')
    cells.append(f'''<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
    </mxCell>''')
    return cid

def add_text(x, y, w, h, value, style):
    """Add a text-only cell."""
    cid = next_id[0]; next_id[0] += 1
    val = html.escape(value).replace('\n', '<br>')
    cells.append(f'''<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
    </mxCell>''')
    return cid

def add_edge(src_id, tgt_id, style, label=''):
    """Add an edge between two cell IDs."""
    cid = next_id[0]; next_id[0] += 1
    val = html.escape(label).replace('\n', '<br>')
    cells.append(f'''<mxCell id="{cid}" value="{val}" style="{style}" edge="1" source="{src_id}" target="{tgt_id}" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>''')
    return cid

# ============================================================
# TITLE
# ============================================================
add_text(20, 10, 1600, 36, "MARST -- Multi-Anchor Residual Spatiotemporal Transformer",
         text_style(fs=22, bold=True))
add_text(20, 46, 1600, 22, "Strictly causal traffic imputation via per-position softmax mixture over classical anchors + residual correction",
         text_style(fs=13, italic=True, color='#444444'))

# ============================================================
# COLUMN 1: INPUTS (x: 30 .. 250)
# ============================================================
add_text(30, 80, 220, 24, "INPUTS", text_style(fs=16, bold=True, italic=True, color='#444'))

inputs = [
    ("X ⊙ M",            "observed values"),
    ("M",                "obs mask"),
    ("HA_prior",         "historical avg lookup"),
    ("A",                "adjacency (0-diag)"),
    ("sin τ, cos τ",     "time of day"),
]
input_ids = []
y0 = 120
for i, (sym, desc) in enumerate(inputs):
    y = y0 + i * 70
    inp = add_box(40, y, 90, 50, sym, box_style(COL_INPUT, COL_INPUT_BD, fs=15, bold=True))
    input_ids.append(inp)
    add_text(135, y, 130, 50, desc, text_style(fs=11, italic=True, color='#333', align='left'))

# Learnable parameters note
add_text(30, 510, 250, 22, "Learnable parameters:", text_style(fs=11, bold=True, color='#555'))
add_text(30, 532, 250, 22, "σ(logit)_n ∈ (0,1) per-sensor decay",
         text_style(fs=10, color='#555'))
add_text(30, 552, 250, 20, "node_emb, pos_emb",
         text_style(fs=10, color='#555'))
add_text(30, 580, 250, 18, "(buffers: node_means/stds, A, adj_2hop — stored)",
         text_style(fs=9, italic=True, color='#888'))

# ============================================================
# COLUMN 2: ANCHORS (x: 290 .. 620)
# ============================================================
add_text(290, 80, 330, 24, "ANCHOR COMPUTATION",
         text_style(fs=16, bold=True, italic=True, color=COL_ANCHOR_BD))

# Three causal anchors group box
anchor_grp = add_box(290, 120, 330, 290, "", box_style(COL_ANCHOR, COL_ANCHOR_BD))
add_text(290, 130, 330, 24, "Three causal anchors",
         text_style(fs=13, bold=True, italic=True, color=COL_ANCHOR_BD))

# a_LOCF
locf_id = add_box(305, 170, 300, 70, "", box_style('#FFF5DD', COL_ANCHOR_BD))
add_text(315, 175, 70, 30, "a_LOCF", text_style(fs=18, bold=True, color=COL_ANCHOR_BD, align='left'))
add_text(395, 178, 200, 22, "soft LOCF (EMA toward HA)",
         text_style(fs=11, bold=True, color='#222', align='left'))
add_text(395, 202, 220, 22, "decay = σ(logit_n)  per-sensor learnable",
         text_style(fs=10, italic=True, color='#5C4500', align='left'))

# a_HA
ha_id = add_box(305, 250, 300, 60, "", box_style('#FFF5DD', COL_ANCHOR_BD))
add_text(315, 257, 70, 30, "a_HA", text_style(fs=18, bold=True, color=COL_ANCHOR_BD, align='left'))
add_text(395, 257, 200, 22, "Historical Average",
         text_style(fs=11, bold=True, color='#222', align='left'))
add_text(395, 280, 200, 22, "per (sensor, slot of day)",
         text_style(fs=10, italic=True, color='#5C4500', align='left'))

# a_KNN
knn_id = add_box(305, 320, 300, 70, "", box_style('#FFF5DD', COL_ANCHOR_BD))
add_text(315, 325, 70, 30, "a_KNN", text_style(fs=18, bold=True, color=COL_ANCHOR_BD, align='left'))
add_text(395, 328, 200, 22, "Spatial Neighbour Mean",
         text_style(fs=11, bold=True, color='#222', align='left'))
add_text(395, 352, 220, 22, "1-hop, mask-aware, HA fallback",
         text_style(fs=10, italic=True, color='#5C4500', align='left'))

# AUX features (separate box)
add_text(290, 425, 330, 22, "AUXILIARY FEATURES (trunk inputs only)",
         text_style(fs=12, bold=True, italic=True, color=COL_AUX_BD))
aux_id = add_box(290, 450, 330, 100, "", box_style(COL_AUX, COL_AUX_BD))
add_text(290, 460, 330, 22, "staleness",
         text_style(fs=12, bold=True, color='#5C4500'))
add_text(290, 480, 330, 18, "(steps since last obs / 48)",
         text_style(fs=9, italic=True, color='#222'))
add_text(290, 502, 330, 22, "2-hop nbr mean of HARD locf, z-scored",
         text_style(fs=11, bold=True, color='#5C4500'))
add_text(290, 524, 330, 18, "(uses HARD locf, NOT soft)",
         text_style(fs=9, italic=True, color='#222'))

# ============================================================
# COLUMN 3: TRUNK (x: 670 .. 1080)
# ============================================================
add_text(670, 80, 410, 24, "SPATIOTEMPORAL TRANSFORMER TRUNK",
         text_style(fs=16, bold=True, italic=True, color=COL_TRUNK_BD))

# Feature stack
fstack_id = add_box(670, 120, 410, 60, "9-channel feature stack  [B, N, T, 9]",
                    box_style(COL_FEAT, COL_FEAT_BD, fs=13, bold=True))
add_text(670, 165, 410, 18, "[x, m, sin τ, cos τ, a_LOCF, a_HA, a_KNN, staleness, 2-hop_locf]",
         text_style(fs=9, italic=True, color='#555'))

# in_proj
inproj_id = add_box(670, 200, 410, 50, "in_proj:  Linear(9 → 128)",
                    box_style('#FFE4A5', '#7A5C00', fs=13, bold=True))

# embeddings
emb_id = add_box(670, 270, 410, 50, "+ node_emb [N, 128]    + pos_emb [T, 128]",
                 box_style('#FFD580', '#5B3F00', fs=12, bold=True))

# STBlock container
stblock_id = add_box(670, 340, 410, 200, "", box_style(COL_TRUNK, COL_TRUNK_BD))
add_text(670, 350, 410, 26, "STBlock", text_style(fs=17, bold=True, color=COL_TRUNK_BD))
add_text(670, 378, 410, 18, "(× 6)", text_style(fs=11, italic=True, color='#444'))

stblk_t = add_box(690, 410, 370, 35, "Causal-Temporal Self-Attention",
                  box_style('#FFFFFF', COL_TRUNK_BD, fs=11.5))
stblk_s = add_box(690, 455, 370, 35, "Mask-Aware Spatial Self-Attention",
                  box_style('#FFFFFF', COL_TRUNK_BD, fs=11.5))
stblk_f = add_box(690, 500, 370, 35, "FFN + GELU + Dropout + LayerNorm",
                  box_style('#FFFFFF', COL_TRUNK_BD, fs=11.5))

# Final LayerNorm
ln_id = add_box(720, 560, 310, 40, "Final LayerNorm",
                box_style('#A5C3DF', '#0F3A66', fs=12, bold=True))

# Hidden state h
h_id = add_box(720, 620, 310, 50, "trunk hidden state  h ∈ R^(B×N×T×128)",
               box_style('#7FA8C9', '#0E3050', fs=12, bold=True, tc='#FFFFFF'))

# ============================================================
# COLUMN 4: HEADS (x: 1130 .. 1450)
# ============================================================
add_text(1130, 80, 320, 24, "OUTPUT HEADS",
         text_style(fs=16, bold=True, italic=True, color='#444'))

# Anchor Gate
gate_id = add_box(1130, 120, 320, 100, "", box_style(COL_PI, COL_PI_BD))
add_text(1130, 130, 320, 26, "Anchor Gate",
         text_style(fs=14, bold=True, color=COL_PI_BD))
add_text(1130, 160, 320, 20, "Linear → GELU → Linear → softmax",
         text_style(fs=11, color='#222'))
add_text(1130, 188, 320, 28, "π(t, n) ∈ Δ²    [3-vector]",
         text_style(fs=15, bold=True, color='#0E4A1B'))

# Residual Head
res_id = add_box(1130, 280, 320, 100, "", box_style(COL_R, COL_R_BD))
add_text(1130, 290, 320, 26, "Residual Head",
         text_style(fs=14, bold=True, color=COL_R_BD))
add_text(1130, 320, 320, 20, "Linear → GELU → Linear",
         text_style(fs=11, color='#222'))
add_text(1130, 348, 320, 28, "r(t, n) ∈ R",
         text_style(fs=15, bold=True, color='#4A1A6A'))

# Meta-Gate
meta_id = add_box(1130, 440, 320, 110, "", box_style(COL_ALPHA, COL_ALPHA_BD))
add_text(1130, 450, 320, 26, "Meta-Gate",
         text_style(fs=14, bold=True, color=COL_ALPHA_BD))
add_text(1130, 480, 320, 20, "Linear([h ; n_obs]) → ... → Sigmoid",
         text_style(fs=11, color='#222'))
add_text(1130, 508, 320, 28, "α(t, n) ∈ (0, 1)",
         text_style(fs=15, bold=True, color='#0A4D44'))

# n_obs box
nobs_id = add_box(810, 720, 280, 50, "n_obs = (A·m) / (A·1)    (nbr obs density)",
                  box_style('#E8F4F1', COL_ALPHA_BD, fs=11.5, bold=True, tc='#0A4D44'))

# ============================================================
# COLUMN 5: COMPOSITION + OUTPUT (x: 1500 .. 1620)
# ============================================================
out_id = add_box(1500, 120, 120, 70, "ŷ",
                 box_style(COL_OUTPUT, COL_OUTPUT_BD, fs=28, bold=True, tc=COL_OUTPUT_BD))
add_text(1500, 190, 120, 18, "output", text_style(fs=11, italic=True, color='#7A1A1A'))

comp_id = add_box(1500, 600, 120, 120, "", box_style(COL_COMP, COL_COMP_BD))
add_text(1500, 610, 120, 22, "compose", text_style(fs=12, bold=True, color=COL_COMP_BD))
add_text(1500, 638, 120, 24, "π · a", text_style(fs=14, color='#7A4500'))
add_text(1500, 662, 120, 24, "+", text_style(fs=14, color='#7A4500'))
add_text(1500, 686, 120, 24, "α · r", text_style(fs=14, color='#7A4500'))

# ============================================================
# ARROWS
# ============================================================

# Raw inputs to anchor compute (use entryX/exitX for proper edge anchors)
for inp in input_ids:
    add_edge(inp, anchor_grp,
             "endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
             f"strokeColor={COL_ARROW};strokeWidth=1.4;endSize=8;")

# Raw inputs to aux features
for inp in input_ids:
    add_edge(inp, aux_id,
             "endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.3;"
             f"strokeColor={COL_ARROW};strokeWidth=1.0;endSize=6;")

# Anchors to feature stack (orange)
add_edge(anchor_grp, fstack_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.3;entryX=0.3;entryY=1;"
         f"strokeColor={COL_ANCHOR_BD};strokeWidth=2.5;endSize=10;curved=1;")

# Aux to feature stack (yellow)
add_edge(aux_id, fstack_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.2;entryX=0.5;entryY=1;"
         f"strokeColor={COL_AUX_BD};strokeWidth=2.0;endSize=10;curved=1;")

# Raw inputs to feature stack (channels 1-4) -- bold blue
add_edge(input_ids[0], fstack_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0.1;entryY=1;"
         f"strokeColor={COL_RAW};strokeWidth=2.4;endSize=10;curved=1;")
add_edge(input_ids[1], fstack_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0.1;entryY=1;"
         f"strokeColor={COL_RAW};strokeWidth=1.5;endSize=8;curved=1;dashed=1;")
add_edge(input_ids[4], fstack_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0.1;entryY=1;"
         f"strokeColor={COL_RAW};strokeWidth=1.5;endSize=8;curved=1;dashed=1;")
add_text(120, 90, 220, 22, "raw inputs (channels 1-4) →",
         text_style(fs=10, bold=True, italic=True, color=COL_RAW))

# Vertical trunk chain
add_edge(fstack_id, inproj_id, edge_style(sw=2.0))
add_edge(inproj_id, emb_id, edge_style(sw=2.0))
add_edge(emb_id, stblock_id, edge_style(sw=2.0))
add_edge(stblock_id, ln_id, edge_style(sw=2.0))
add_edge(ln_id, h_id, edge_style(sw=2.0))

# h to the three heads
add_edge(h_id, gate_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
         f"strokeColor={COL_ARROW};strokeWidth=2.0;endSize=10;curved=1;")
add_edge(h_id, res_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
         f"strokeColor={COL_ARROW};strokeWidth=2.0;endSize=10;curved=1;")
add_edge(h_id, meta_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.3;"
         f"strokeColor={COL_ARROW};strokeWidth=2.0;endSize=10;curved=1;")

# n_obs to meta-gate (second input)
add_edge(nobs_id, meta_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.7;"
         f"strokeColor={COL_ALPHA_BD};strokeWidth=1.8;endSize=10;curved=1;")

# Anchor BYPASS arrow to composition
add_edge(anchor_grp, comp_id,
         f"endArrow=classic;html=1;exitX=1;exitY=1;entryX=0;entryY=0.5;"
         f"strokeColor={COL_BYPASS};strokeWidth=3.0;endSize=12;curved=1;",
         label="anchors [a_LOCF, a_HA, a_KNN]\\n(bypass trunk, multiplied by π at composition)")

# Heads to composition
add_edge(gate_id, comp_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.2;"
         f"strokeColor={COL_PI_BD};strokeWidth=2.0;endSize=10;curved=1;",
         label="π")
add_edge(res_id, comp_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
         f"strokeColor={COL_R_BD};strokeWidth=2.0;endSize=10;curved=1;",
         label="r")
add_edge(meta_id, comp_id,
         f"endArrow=classic;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.8;"
         f"strokeColor={COL_ALPHA_BD};strokeWidth=2.0;endSize=10;curved=1;",
         label="α")

# Composition to output
add_edge(comp_id, out_id,
         f"endArrow=classic;html=1;exitX=0.5;exitY=0;entryX=0.5;entryY=1;"
         f"strokeColor={COL_OUTPUT_BD};strokeWidth=3.0;endSize=14;")

# ============================================================
# BOTTOM EQUATION
# ============================================================
add_text(20, 820, 1620, 36,
         "ŷ(t, n) = π_LOCF(t,n) · a_LOCF(t,n) + π_HA(t,n) · a_HA(t,n) + π_KNN(t,n) · a_KNN(t,n) + α(t,n) · r(t,n)",
         text_style(fs=15, color='#111'))

# ============================================================
# WRITE FILE
# ============================================================
cells_xml = "\n".join(cells)
doc = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" type="device" version="24.7.0">
  <diagram name="MARST Architecture" id="marst-arch">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1700" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{cells_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''

out_path = r'C:/Users/new/Documents/redfinal/models/MARST_architecture.drawio'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(doc)

import os
print(f"Saved {out_path} ({os.path.getsize(out_path)} bytes)")
print(f"Cells written: {len(cells)}")

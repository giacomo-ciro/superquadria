# Superquadric Furniture Assemblies Reference

This document provides normalized geometric templates for multi-primitive furniture assemblies built entirely from superquadrics.

> **Note on Agent Usage:** These templates are reference archetypes illustrating how to compose superquadric primitives (combining boxes, cylinders, and rounded cushions) into realistic composite furniture. They are intended as geometric references—not a rigid enum. Agents should adapt dimensions, orientations, and create novel assemblies matching each room's specific style and brief.

---

## 1. Coordinate & Normalization Convention

Every assembly is normalized inside a **unit bounding box**:
- **$X$ (Width):** $[-0.5, 0.5]$ (total width extent $= 1.0$, centered at $X = 0$)
- **$Y$ (Height):** $[0.0, 1.0]$ (total height extent $= 1.0$, resting on the base at $Y = 0$)
- **$Z$ (Depth):** $[-0.5, 0.5]$ (total depth extent $= 1.0$, centered at $Z = 0$)

### Instantiating in World Space
To place an assembly at room position $(P_x, P_y, P_z)$ with desired physical dimensions $(W, H, D)$ in meters:
- **Primitive Center:**
  $$\text{position} = [P_x + \text{pos}_{\text{norm}}[0] \times W,\; P_y + \text{pos}_{\text{norm}}[1] \times H,\; P_z + \text{pos}_{\text{norm}}[2] \times D]$$
- **Primitive Scale (Semiaxis):**
  - For unrotated primitives (`rotation: [0, 0, 0]`):
    $$\text{scale} = [\text{scale}_{\text{norm}}[0] \times W,\; \text{scale}_{\text{norm}}[1] \times H,\; \text{scale}_{\text{norm}}[2] \times D]$$
  - For cylinders standing upright (`rotation: [90, 0, 0]` where local $Z$ maps to world $Y$):
    $$\text{scale} = [\text{scale}_{\text{norm}}[0] \times W,\; \text{scale}_{\text{norm}}[1] \times D,\; \text{scale}_{\text{norm}}[2] \times H]$$

---

## 2. Normalized Assembly Templates

### 1. Dining Table (`dining_table`)
- **Description:** Rectangular tabletop with 4 corner legs (5 primitives).
- **Typical Dimensions:** $W = 1.80\text{m},\; H = 0.75\text{m},\; D = 1.10\text{m}$.

```json
[
  {"assembly": "dining_table", "position": [0.0, 0.96, 0.0], "scale": [0.5, 0.04, 0.5], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_table", "position": [0.417, 0.467, 0.382], "scale": [0.022, 0.467, 0.036], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_table", "position": [-0.417, 0.467, 0.382], "scale": [0.022, 0.467, 0.036], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_table", "position": [0.417, 0.467, -0.382], "scale": [0.022, 0.467, 0.036], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_table", "position": [-0.417, 0.467, -0.382], "scale": [0.022, 0.467, 0.036], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 2. Pedestal Table (`pedestal_table`)
- **Description:** Round table with central column and disc base (3 primitives).
- **Typical Dimensions:** $W = 1.20\text{m},\; H = 0.75\text{m},\; D = 1.20\text{m}$.

```json
[
  {"assembly": "pedestal_table", "position": [0.0, 0.96, 0.0], "scale": [0.5, 0.5, 0.04], "exponents": [0.1, 1.0], "rotation": [90.0, 0.0, 0.0]},
  {"assembly": "pedestal_table", "position": [0.0, 0.48, 0.0], "scale": [0.05, 0.05, 0.44], "exponents": [0.1, 1.0], "rotation": [90.0, 0.0, 0.0]},
  {"assembly": "pedestal_table", "position": [0.0, 0.027, 0.0], "scale": [0.292, 0.292, 0.027], "exponents": [0.1, 1.0], "rotation": [90.0, 0.0, 0.0]}
]
```

---

### 3. Dining Chair (`dining_chair`)
- **Description:** Chair with cushioned seat, backrest, and base support (3 primitives).
- **Typical Dimensions:** $W = 0.44\text{m},\; H = 0.88\text{m},\; D = 0.44\text{m}$.

```json
[
  {"assembly": "dining_chair", "position": [0.0, 0.511, 0.0], "scale": [0.5, 0.034, 0.5], "exponents": [0.2, 0.2], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_chair", "position": [0.0, 0.773, -0.432], "scale": [0.455, 0.227, 0.068], "exponents": [0.2, 0.2], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "dining_chair", "position": [0.0, 0.239, 0.0], "scale": [0.409, 0.239, 0.409], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 4. Bed (`bed`)
- **Description:** Base frame, rounded mattress, headboard, and pillows (4 primitives).
- **Typical Dimensions:** $W = 1.96\text{m},\; H = 1.00\text{m},\; D = 2.10\text{m}$.

```json
[
  {"assembly": "bed", "position": [0.0, 0.15, 0.0], "scale": [0.485, 0.15, 0.5], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bed", "position": [0.0, 0.40, 0.024], "scale": [0.459, 0.12, 0.476], "exponents": [0.4, 0.4], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bed", "position": [0.0, 0.55, -0.467], "scale": [0.5, 0.45, 0.029], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bed", "position": [0.0, 0.57, -0.333], "scale": [0.357, 0.06, 0.095], "exponents": [0.5, 0.5], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 5. Armchair (`armchair`)
- **Description:** Rounded seat cushion, backrest, and two padded armrests (4 primitives).
- **Typical Dimensions:** $W = 1.02\text{m},\; H = 0.76\text{m},\; D = 0.76\text{m}$.

```json
[
  {"assembly": "armchair", "position": [0.0, 0.237, 0.0], "scale": [0.373, 0.237, 0.5], "exponents": [0.3, 0.3], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "armchair", "position": [0.0, 0.632, -0.368], "scale": [0.373, 0.368, 0.132], "exponents": [0.3, 0.3], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "armchair", "position": [-0.422, 0.408, 0.0], "scale": [0.078, 0.263, 0.5], "exponents": [0.3, 0.3], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "armchair", "position": [0.422, 0.408, 0.0], "scale": [0.078, 0.263, 0.5], "exponents": [0.3, 0.3], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 6. Bookshelf (`bookshelf`)
- **Description:** Tall outer cabinet casing with 3 storage shelves (4 primitives).
- **Typical Dimensions:** $W = 1.10\text{m},\; H = 1.80\text{m},\; D = 0.40\text{m}$.

```json
[
  {"assembly": "bookshelf", "position": [0.0, 0.50, 0.0], "scale": [0.5, 0.5, 0.5], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bookshelf", "position": [0.0, 0.278, 0.05], "scale": [0.455, 0.011, 0.45], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bookshelf", "position": [0.0, 0.528, 0.05], "scale": [0.455, 0.011, 0.45], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "bookshelf", "position": [0.0, 0.778, 0.05], "scale": [0.455, 0.011, 0.45], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 7. Kitchen Counter (`kitchen_counter`)
- **Description:** Base cabinetry unit, countertop slab, and cooktop hob (3 primitives).
- **Typical Dimensions:** $W = 2.08\text{m},\; H = 0.94\text{m},\; D = 0.76\text{m}$.

```json
[
  {"assembly": "kitchen_counter", "position": [0.0, 0.468, 0.0], "scale": [0.481, 0.468, 0.461], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "kitchen_counter", "position": [0.0, 0.957, 0.0], "scale": [0.5, 0.021, 0.5], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]},
  {"assembly": "kitchen_counter", "position": [-0.192, 0.989, 0.0], "scale": [0.154, 0.011, 0.368], "exponents": [0.1, 0.1], "rotation": [0.0, 0.0, 0.0]}
]
```

---

### 8. Floor Lamp (`floor_lamp`)
- **Description:** Weighted disc base, vertical stem rod, and lampshade (3 primitives).
- **Typical Dimensions:** $W = 0.44\text{m},\; H = 1.70\text{m},\; D = 0.44\text{m}$.

```json
[
  {"assembly": "floor_lamp", "position": [0.0, 0.012, 0.0], "scale": [0.5, 0.5, 0.012], "exponents": [0.1, 1.0], "rotation": [90.0, 0.0, 0.0]},
  {"assembly": "floor_lamp", "position": [0.0, 0.441, 0.0], "scale": [0.057, 0.057, 0.424], "exponents": [0.1, 1.0], "rotation": [90.0, 0.0, 0.0]},
  {"assembly": "floor_lamp", "position": [0.0, 0.912, 0.0], "scale": [0.455, 0.455, 0.088], "exponents": [0.2, 1.0], "rotation": [90.0, 0.0, 0.0]}
]
```

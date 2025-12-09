## 📁 Dataset Structure

The raw microscopy dataset follows a strict hierarchical organization defined by the laboratory workflow.  
Each patient corresponds to one cultured sample, and each sample is imaged in **five wells** and **five spatial zones** (A–E), in **two acquisition modes** (RGB and BW).

The filesystem structure is:

```
HPMCs_DialisePeritoneal/
│
├── 1056/                        # Patient ID
│   ├── Poço 1/
│   │   ├── 1.A.1.jpg            # Well 1 – Zone A – Mode 1 (RGB)
│   │   ├── 1.A.2.jpg            # Well 1 – Zone A – Mode 2 (BW)
│   │   ├── 1.B.1.jpg
│   │   ├── 1.B.2.jpg
│   │   ├── 1.C.1.jpg
│   │   ├── 1.C.2.jpg
│   │   ├── 1.D.1.jpg
│   │   ├── 1.D.2.jpg
│   │   ├── 1.E.1.jpg
│   │   ├── 1.E.2.jpg
│   │   ...
│   ├── Poço 2/
│   ├── Poço 3/
│   ├── Poço 4/
│   └── Poço 5/
│
├── 1059/
├── 1060/
├── 1062/
├── 1065/
├── 1066/
├── 1067/
├── 1068/
└── 1069/
```

### ✔ Meaning of the naming convention

Each filename follows the laboratory’s acquisition protocol:

```
{well}.{zone}.{mode}.jpg
```

Where:

- **well** ∈ {1, 2, 3, 4, 5}  
- **zone** ∈ {A, B, C, D, E}  
- **mode**  
  - **1 → RGB** (colour image of the same field)  
  - **2 → BW** (high-contrast grayscale for cell-body and nuclear visibility)

Thus, each well contributes:

- **5 Zones × 2 Modes = 10 images per well**
- **5 Wells × 10 = 50 images per patient**

All analyses implemented in the project assume and validate this structure.

---

## 🔍 How This Structure Is Loaded in the Pipeline

Every feature-extraction script relies on this organization.  
All scripts begin by scanning the directory recursively and parsing the identifiers directly from filenames:

```
patient_id / well / zone / mode
```

For example, in:

```
1060/Poço 3/3.C.2.jpg
```

We extract:

| Component     | Meaning                     |
|--------------|-----------------------------|
| 1060         | Patient                     |
| Poço 3       | Well                        |
| 3            | Well ID (redundancy check)  |
| C            | Spatial zone                |
| 2            | BW acquisition              |

This guarantees reproducibility and allows your code to:

- validate file structure  
- detect missing images  
- join with clinical metadata  
- aggregate by well, zone, and patient  
- compute patient-level summaries  
- cluster in meaningful hierarchical units  

---

## 📌 Why This Structure Must Stay Private

The images belong to real patients and contain identifiable biological material.  
Thus:

- **Raw images cannot be committed to GitHub**  
- **Repository must be PRIVATE**  
- Only extracted numerical features are uploaded (safe & anonymised)


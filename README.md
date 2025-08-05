# 🎨 Neural Style Transfer (NST)

Neural Style Transfer (NST) is a deep learning technique that **combines the content of one image with the artistic style of another** to create a unique, blended output.

Think:  
> *"What if your selfie was painted by Van Gogh?"* 🖌️✨

---

## 📖 How It Works

NST uses a **pre-trained Convolutional Neural Network (CNN)** (often **VGG19**) to extract features from images:

1. **Content Representation** 🏙️  
   - Captures *what* is in the image (shapes, structures, objects).
   - Extracted from **deeper layers** of the CNN.

2. **Style Representation** 🎨  
   - Captures *how* the image looks (textures, brush strokes, color patterns).
   - Extracted from **multiple shallow + deep layers** via **Gram Matrices**.

3. **Optimization** ⚙️  
   - Start from a random/noise image.
   - Iteratively update pixels to **minimize**:
     - **Content Loss** – difference in content features between generated and content image.
     - **Style Loss** – difference in style features between generated and style image.
   - The balance between these losses is controlled by weighting factors.

---

## 🧮 Core Equations

**Total Loss:**
\[
L_{\text{total}} = \alpha \cdot L_{\text{content}} + \beta \cdot L_{\text{style}}
\]

- **α (alpha)** → weight for content preservation.
- **β (beta)** → weight for style patterns.

---

## 📂 Project Structure


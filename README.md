# 🎨Aseprite AI Generator (AAG)

**Aseprite AI Generator (AAG)** คือสะพานเชื่อมระหว่างพลังของ Generative AI (Stable Diffusion) เข้ากับเครื่องมือวาด Pixel Art Aseprite โปรแกรมนี้จะเปลี่ยนข้อความของคุณให้กลายเป็นสไปรท์ที่พร้อมใช้งานในเกมภายในไม่กี่วินาทีโดยประมวลผลผ่าน GPU ในเครื่องของคุณ 100%

---

### 🌟 Key Highlights

* **🔒 Local & Private:** ข้อมูลทั้งหมดประมวลผลในเครื่องคุณ ไม่มีการส่งรูปภาพไปยัง Cloud ความลับโปรเจกต์ปลอดภัยแน่นอน
* **📐 Professional Pixel Quantization:** ใช้อัลกอริทึม Median Cut และ Nearest Neighbor เพื่อแปลงภาพ AI ให้เป็นพิกเซลที่คมกริบ พร้อมจัดการจำนวนสี (Palette) ได้ตามต้องการ
* **🎭 AI Background Removal:** ขับเคลื่อนด้วย **BiRefNet** โมเดลลบพื้นหลังระดับ High-end ที่ช่วยลบพื้นหลังที่ซับซ้อนออกให้อัตโนมัติ ได้ไฟล์ PNG โปร่งใสทันที
* **🎨 LoRA Support:** รองรับการใช้งาน LoRA เพื่อคุมสไตล์งานให้เฉพาะเจาะจง เช่น 8-bit, 16-bit, Isometric หรือสไตล์ศิลปินที่ชื่นชอบ
* **🚀 Optimized for SDXL:** รองรับทั้ง Stable Diffusion XL (คุณภาพสูง) และ SD 1.5 (เน้นความเร็ว)

### 🚀 Upgrade Highlights

*   **⚡ Next-Gen GPU Support:** รองรับสถาปัตยกรรม GPU รุ่นล่าสุด (รวมถึง **RTX 5070 Ti**) เพื่อการประมวลผลที่รวดเร็วระดับวินาที
*   **🎭 Advanced BG Removal:** เพิ่มระบบ **BiRefNet** เพื่อการลบพื้นหลังที่แม่นยำกว่าเดิม ให้สไปรท์ที่พร้อมใช้งานทันที
*   **🧠 Memory Optimization:** ปรับปรุงการจัดการ VRAM ใหม่ทั้งหมด ทำให้รันโมเดลใหญ่อย่าง SDXL บนการ์ดจอรุ่นกลางได้อย่างลื่นไหล
*   **🎨 SDXL LoRA Integration:** รองรับการใช้ LoRA บนฐาน SDXL เพื่อคุมสไตล์ Pixel Art ให้เฉพาะเจาะจงและคมชัดกว่าเดิม
*   **💎 Sharp Color Quantization:** อัลกอริทึมคัดกรองสีเวอร์ชันใหม่ที่ช่วยรักษาขอบพิกเซลให้คมกริบ (Clean Edge) และโทนสีที่สมจริง

---

### 📋 Prerequisites
ก่อนติดตั้งโปรดตรวจสอบว่าเครื่องของคุณมีสเปกดังนี้:
* **OS:** Windows 10/11
* **Python:** 3.10.11 (แนะนำ)
* **GPU:** NVIDIA GPU พร้อม VRAM 8GB+ (เพื่อประสิทธิภาพสูงสุด)
* **Storage:** พื้นที่ว่าง 10GB+
* **Aseprite:** v1.2.10 หรือเวอร์ชันที่ใหม่กว่า

---

### 📊 Technical Specifications
| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.10.11 / Flask |
| **AI Engine** | Diffusers (SDXL / SD 1.5) |
| **Segmentation** | BiRefNet (zhengpeng7/BiRefNet) |
| **Processing** | PyTorch / Torchvision / PIL |
| **Interface** | REST API (JSON / Base64) |

---

### 🚀 Getting Started

1.  **Clone this repo:**
    ```bash
    git clone https://github.com/TangTaksin/local-aseprite-ai-generator.git
    ```

2.  **Run Start Server.bat:**
    ดับเบิลคลิกไฟล์ `Start Server.bat` ระบบจะทำการสร้าง Virtual Environment และติดตั้ง Library ที่จำเป็นให้อัตโนมัติ

3.  **Setup Model & Network (หน้าจอ Interactive):**
    เมื่อหน้าต่าง CMD ปรากฏขึ้น ให้ทำตามขั้นตอนดังนี้:
    *   **Base Model Selection:** เลือกโมเดลเริ่มต้น (แนะนำเลข `1` สำหรับ SDXL หรือ `2` สำหรับเครื่องที่สเปกไม่สูงมาก)
    *   **Network Configuration:** สำหรับการรันครั้งแรก ให้เลือก **Online (N)** เพื่อดาวน์โหลดโมเดลจากอินเทอร์เน็ต

4.  **Ready to Go!:**
    เมื่อดาวน์โหลดสำเร็จ ระบบจะโหลดโมเดลเข้า GPU และขึ้นข้อความสีเขียวว่า **✅ Server ready!** 
    > **⚠️ ข้อควรระวัง:** ห้ามปิดหน้าต่าง CMD ในขณะใช้งาน เพราะนี่คือส่วนประมวลผลหลักที่คอยรับคำสั่งจาก Aseprite

---

### 🛠 How to Use in Aseprite

1.  **ติดตั้ง Plugin:** นำไฟล์ .aseprite-extension ไปติดตั้งใน Aseprite ผ่านเมนู `Edit > Preferences > Extensions > Add Extension`
2.  **เปิดใช้งาน:** ไปที่เมนู `File > Local AI Generator`
3.  **ใส่ Prompt:** พิมพ์คำอธิบายภาพที่ต้องการ (เช่น *cat warrior, sword, blue armor*)
4.  **กด Generate:** ภาพจะถูกสร้างและนำมาวางบน Canvas ของคุณโดยอัตโนมัติ!

---

### 🙏 Acknowledgments & Credits

โปรเจกต์นี้เป็นการพัฒนาต่อยอด (Modified & Enhanced) จากผลงานต้นฉบับ:
* **Original Creator:** [Red335](https://red335.itch.io/pixelai-local-ai-directly-in-aseprite) ผู้พัฒนา **PixelAI**



---

### 🛡 License

Distributed under the **MIT License**. See `LICENSE` for more information.

Copyright (c) 2026 **TangTaksin**
```
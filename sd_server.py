#!/usr/bin/env python3
"""
เซิร์ฟเวอร์ Local AI Generator สำหรับ Aseprite
ไฟล์นี้ทำหน้าที่เป็น Entry point หลักเพื่อส่งต่อการรันไปยังโมดูลย่อยใน src/
"""

import sys
from typing import Optional

# นำทางไปรันเซิร์ฟเวอร์จากโมดูลหลักที่ถูกแยกส่วน
try:
    from src.api_server import main as run_server
except ImportError:
    # เผื่อโครงสร้าง path มีปัญหา ให้เพิ่มโฟลเดอร์ปัจจุบันเข้าไป
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.api_server import main as run_server

def main(default_model_to_load: Optional[str] = None, offline: bool = False) -> None:
    run_server(default_model_to_load=default_model_to_load, offline=offline)

if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")
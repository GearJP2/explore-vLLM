# Explore vLLM — Learning & Contribution Roadmap

บันทึกการเรียนรู้ vLLM และ llm-d ผ่านการอ่านโค้ด ทดลอง และ contribute
เริ่มจาก CLI configuration แล้วค่อยเข้าไปถึง inference, KV cache และระบบ serving หลายเครื่อง
ใช้ CPU เป็นหลัก และใช้ cloud GPU เมื่อจำเป็นต้องตรวจพฤติกรรมจริง

อัปเดตบันทึก: 12 ตุลาคม 2026

## ตอนนี้เราอยู่ตรงไหน

**กำลังศึกษา P/D disaggregation และ request draining ก่อนเลือกขอบเขตงานใน llm-d #1525**

PR แรกเขียน ทดสอบ และเปิดส่งแล้ว ส่วนงาน llm-d ยังอยู่ช่วงเรียนรู้และตามโค้ด
ยังไม่ได้ยืนยันบั๊กหรือเริ่ม patch สำหรับ P/D shutdown
สถานะในหน้านี้เป็นความคืบหน้าของเรา ไม่ใช่สถานะ review/merge แบบสดจาก upstream

📖 เริ่มอ่าน: [คู่มือภาษาไทย — พื้นฐานก่อนทำ request draining](docs/llm-d-request-draining-study.md)

## Roadmap

| ช่วง | สิ่งที่เรียน/ทำ | ความคืบหน้า |
| --- | --- | --- |
| 1. รู้จัก repo และเลือกงาน | อ่าน issue, เช็กงานซ้ำ, เลือกงานที่ทดลองบน CPU ได้ | ทำแล้ว |
| 2. CLI และ configuration | FrontendArgs → parser → args → consumers, YAML/env precedence | เรียนและใช้ใน PR แรกแล้ว |
| 3. Contribution แรก | เพิ่ม HTTP keep-alive CLI config พร้อม compatibility และ tests | เปิด PR แล้ว |
| 4. พื้นฐาน inference | Tensor, weights, prefill/decode, KV cache, TP และ replicas | เรียนเบื้องต้นแล้ว กำลังทบทวน |
| 5. ระบบ P/D และ shutdown | Deployment, KV transfer, cache lifetime และ signal handling | กำลังอ่านโค้ด |
| 6. ระบุงานถัดไป | ตรวจ version, งานที่มีคนทำแล้ว และหาช่องว่างที่พิสูจน์ได้ | ยังต้องทำ |
| 7. ทดลองและ contribute | Reproduce → patch → validation → review → PR | หลังยืนยันขอบเขตงาน |

## Contribution แรก: HTTP keep-alive configuration

- Issue หลัก: [vLLM #25700](https://github.com/vllm-project/vllm/issues/25700)
- PR ของเรา: [vLLM #60459](https://github.com/vllm-project/vllm/pull/60459)
- Implementation commit: [`690946a`](https://github.com/GearJP2/vllm/commit/690946a0145a0940be5748b89b7760d500e5711b)

สิ่งที่ทำแล้ว:

- [x] ไล่ `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` และ consumers ที่เกี่ยวข้อง
- [x] เพิ่ม `--http-timeout-keep-alive` ผ่าน CLI/YAML
- [x] ส่งค่าที่ resolve แล้วไป Python API server, render server และ Rust frontend bootstrap
- [x] เก็บ env fallback เพื่อรองรับ deployment เดิม โดย CLI > YAML > env > default 5 วินาที
- [x] รองรับค่า 0 และปฏิเสธค่าติดลบ
- [x] รัน focused CPU tests: 9 passed และ pre-commit ผ่าน
- [x] เปิด PR และเปิดเผยการใช้ AI ช่วยทำงาน

สิ่งที่ได้เรียน: argument parsing, config precedence, การตาม consumer,
การส่ง Python config ไป Rust ผ่าน JSON และ backward compatibility
ขอบเขตสุดท้ายยังไม่ได้ประกาศ deprecation หรือลบ env var

เก็บ [แผนก่อนเริ่มงาน](docs/http-keep-alive-original-plan.md) ไว้อ่านย้อนหลัง
ข้อความสถานะและข้อเสนอในแผนเป็นข้อมูลตอนเตรียมงาน ไม่ใช่ผลลัพธ์สุดท้าย

## หัวข้อปัจจุบัน: llm-d request draining

- Issue ที่ศึกษา: [llm-d #1525](https://github.com/llm-d/llm-d/issues/1525)
- งานก่อนหน้าที่ใช้อ่านประกอบ: [PR #2601](https://github.com/llm-d/llm-d/pull/2601)
- ส่วนที่เลือกสำรวจ: `pd-disaggregation` — แยก prefill และ decode

คำถามหลัก: **ถ้า prefill คำนวณจบ แต่ decode ยังอ่าน KV cache ไม่เสร็จ ระบบรออย่างไรก่อนปิด?**

```text
อ่าน deployment และ image
    ↓
เข้าใจ prefill → KV transfer → decode
    ↓
ตามการถือและคืน KV blocks
    ↓
ตาม SIGTERM → engine shutdown → connector cleanup
    ↓
ตรวจว่ามีช่องว่างจริง แล้วออกแบบการทดสอบ
```

สิ่งที่ทำแล้ว:

- [x] ศึกษาความหมาย tensor, shape, weights และ tensor parallelism
- [x] แยก prefill/decode และเข้าใจว่า KV cache ใช้ต่อระหว่าง decode
- [x] ศึกษา graceful shutdown, SIGTERM, preStop และ termination grace period
- [x] อ่านตัวอย่าง P/D config: producer/consumer, TP และ routing proxy
- [x] อ่าน `request_finished()` และแนวคิดเลื่อนการคืน KV blocks
- [x] รวบรวมแผนอ่านโค้ดและคำถามทบทวนในคู่มือ

ขั้นถัดไป:

- [ ] อ่านคู่มือบท 2–5 และลองตอบคำถามโดยไม่เปิดเฉลย
- [ ] ตามว่า worker รายงาน `finished_sending` เมื่อไร รวมกรณี timeout/abort
- [ ] ตามว่า engine shutdown นับงาน KV transfer ที่ค้างอยู่หรือไม่
- [ ] ตรวจ vLLM image/version ของ overlay ให้ตรงกับโค้ดที่ศึกษา
- [ ] ตรวจ manifests หลังรวม Kustomize และลำดับปิด proxy/modelserver
- [ ] เช็ก PR ที่เกี่ยวข้องใหม่และตกลงส่วนงานที่ยังขาดก่อนเริ่ม contribution
- [ ] ออกแบบการทดลอง CPU/GPU ตามพฤติกรรมที่ต้องพิสูจน์

## เอกสารใน repo

| เอกสาร | ใช้ทำอะไร |
| --- | --- |
| [คู่มือ P/D และ request draining](docs/llm-d-request-draining-study.md) | อ่านพื้นฐาน ไล่โค้ด และทำคำถามทบทวนพร้อมเฉลย |
| [แผน HTTP keep-alive เดิม](docs/http-keep-alive-original-plan.md) | เก็บเหตุผลและการสำรวจตอนเริ่ม contribution แรก |
| [ร่าง issue comment](issue-comment-draft.md) | บันทึกร่างข้อความสำหรับงานแรก |

## วิธีเรียนและทำงานร่วมกัน

ตั้งคำถามหนึ่งข้อ → หาโค้ดที่เกี่ยวข้อง → อ่านช่วงสั้น ๆ → อธิบายด้วยภาษาตัวเอง
→ ทดลองยืนยัน → ค่อยเสนอการแก้ไข

ให้ AI ช่วยค้นและอธิบายได้ แต่ก่อนส่ง PR ต้องอธิบายได้ว่าทำไมแก้ตรงนี้
พฤติกรรมก่อน/หลังต่างกันอย่างไร และการทดสอบพิสูจน์อะไร
การมีฟังก์ชันชื่อ `shutdown` ไม่ได้ยืนยันว่ารองานทุกประเภทครบแล้ว

`vllm/` เป็น checkout แยกที่ถูก ignore; clone repo นี้อย่างเดียวจะไม่ได้ source vLLM มาด้วย
คู่มือมีลิงก์ GitHub ที่ตรึง revision เพื่อเปิดอ่านได้โดยไม่ต้องมี checkout ในเครื่อง

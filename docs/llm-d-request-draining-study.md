# เรียนรู้ก่อนทำ llm-d #1525: P/D disaggregation และ request draining

เอกสารสำหรับอ่านด้วยตัวเองจากพื้นฐาน ไปจนถึงการตั้งคำถามกับโค้ดจริง
เป้าหมายคืออธิบายได้ว่า “เมื่อปิด prefill หรือ decode งานและ KV cache ที่ค้างอยู่จะเป็นอย่างไร”
ยังไม่ใช่ข้อเสนอแก้ไขหรือข้อสรุปว่า shutdown มีบั๊ก

## 1. โจทย์ที่เรากำลังศึกษา

[llm-d #1525](https://github.com/llm-d/llm-d/issues/1525)
ขอให้ปรับ guides เพื่อกำหนด request draining ให้เหมาะสม
ส่วน [PR #2601](https://github.com/llm-d/llm-d/pull/2601)
เป็นงานก่อนหน้าที่เราใช้ศึกษาแนวทางและขอบเขตที่ยังไม่ได้ครอบคลุม
ก่อนเริ่ม contribution ต้องอ่านสถานะ ความเห็น และ PR ที่เกี่ยวข้องใหม่เสมอ
issue ที่ยังเปิดไม่ได้แปลว่างานทุกส่วนยังว่าง

เราเลือก **P/D disaggregation เป็นหัวข้อสำรวจ** เพราะต้องเข้าใจทั้งการสร้าง token
การส่ง KV cache และการปิดหลาย process ที่พึ่งพากัน
การอ่านและวิเคราะห์ทำบน CPU ได้ ส่วนการยืนยันพฤติกรรม GPU transfer ต้องใช้ deployment ที่ตรงกับงาน

แนะนำให้อ่านสามรอบ:

1. รอบแรก: บท 2–5 เข้าใจโมเดลและ cache
2. รอบสอง: บท 6–8 เข้าใจ deployment และ shutdown
3. รอบสาม: บท 9–12 เปิดโค้ดตามแล้วตอบคำถามด้วยหลักฐาน

## 2. Tensor, shape และ weights

Tensor คือข้อมูลตัวเลขที่จัดเป็นมิติต่าง ๆ:

| ตัวอย่าง | จำนวนมิติ | Shape |
| --- | --- | --- |
| `5` | 0 | `()` |
| `[2, 4, 6]` | 1 | `(3,)` |
| `[[1, 2, 3], [4, 5, 6]]` | 2 | `(2, 3)` |
| 2 ตาราง แต่ละตารางมี 4 แถว 8 คอลัมน์ | 3 | `(2, 4, 8)` |

Shape บอกขนาดแต่ละมิติ ไม่ได้บอกความหมาย ต้องดูบริบทของโค้ดด้วย
เช่น `(4, 8)` อาจหมายถึง 4 tokens แต่ละ token มี representation 8 ค่า
นี่เป็นตัวอย่างเพื่ออธิบาย ไม่ใช่ขนาดจริงของโมเดลที่ใช้ใน guide

Weights คือพารามิเตอร์ที่เรียนรู้มาระหว่างการฝึกและถูกโหลดไว้เพื่อคำนวณ inference
ส่วน activations คือผลคำนวณระหว่างชั้น ทั้งสองอย่างเก็บเป็น tensor ได้
KV cache ก็เป็น tensor แต่ไม่ใช่ weights และไม่ใช่คลังคำตอบสำเร็จรูป

## 3. โมเดลสร้างคำตอบอย่างไร

```text
ข้อความ → tokenizer → token IDs → embedding → ชั้นต่าง ๆ ของโมเดล
                                                      ↓
                                               logits (คะแนน)
                                                      ↓
                                           เลือก token ถัดไป
                                                      ↓
                                           นำไปคำนวณรอบถัดไป
```

Token อาจเป็นคำ ส่วนของคำ เครื่องหมาย หรือหน่วยอื่นตาม tokenizer
อย่าสมมติว่า 1 token เท่ากับ 1 คำ
Logits เป็นคะแนนก่อนแปลงเป็นความน่าจะเป็น การเลือก token ขึ้นอยู่กับ sampling settings ด้วย

**Prefill** ประมวลผล prompt เพื่อสร้างสถานะสำหรับสร้างคำตอบต่อ
ผลจาก prefill โดยทั่วไปใช้เลือก output token แรกได้ด้วย
**Decode** ประมวลผลต่อเนื่องเพื่อสร้าง token ถัด ๆ ไป
รายละเอียดการแบ่งขั้นอาจต่างกันเมื่อมี speculative decoding หรือ optimization อื่น

## 4. KV cache สร้างเมื่อไร อยู่ที่ไหน และใช้ทำอะไร

ใน attention มีค่า Query, Key และ Value
สำหรับการเรียนรอบนี้ ให้เข้าใจว่า token ใหม่ใช้ Query ของตัวเองร่วมกับ Key/Value
ของบริบทที่มีอยู่ เพื่อคำนวณ representation ที่คำนึงถึงข้อความก่อนหน้า

KV cache เก็บ Key/Value ของ tokens ที่ประมวลผลแล้วในชั้น attention
จึงไม่ต้องสร้าง K/V ของ tokens เก่าซ้ำทุกครั้ง แต่ยังมีงาน attention
และงานของชั้นอื่น ๆ ทุกครั้งที่ประมวลผล token ใหม่

```text
Prefill: สร้าง KV ของ prompt
    ↓
Decode: ใช้ KV เดิม + เพิ่ม KV ของ token ที่ประมวลผลใหม่
    ↓
Decode: ใช้ cache ที่สะสมมา + เพิ่มข้อมูลต่อ
    ↓
Request จบ: คืนพื้นที่ หรือเก็บบางส่วนใช้ซ้ำตามนโยบาย cache
```

Phase คือช่วงการทำงาน ไม่ใช่สถานที่เก็บข้อมูล
โดยทั่วไป KV ที่ใช้คำนวณอยู่ในหน่วยความจำอุปกรณ์ แต่มีระบบ offload/transfer ได้ด้วย

Cache ถูกจัดเป็น **blocks** หรือก้อนพื้นที่ เพื่อจัดสรรและนำกลับมาใช้ได้
`block_ids` คือหมายเลขอ้างอิงก้อนเหล่านั้น
`free blocks` หมายถึงคืนสิทธิ์ใช้พื้นที่ ไม่จำเป็นต้องเขียนเลขศูนย์ทับข้อมูลทันที
หากนำพื้นที่ไปใช้กับ request ใหม่ขณะที่อีกฝั่งยังอ่านอยู่ ข้อมูลอาจถูกเขียนทับ

## 5. P/D disaggregation และ tensor parallelism

แบบทั่วไป prefill และ decode อยู่ใน instance เดียวกัน
แบบแยก P/D มี instance สำหรับแต่ละบทบาท และต้องโอน KV ระหว่างกัน

```text
Prefill instance                 Decode instance
โหลด weights ของโมเดล           โหลด weights ของโมเดล
คำนวณ prompt                    รับ KV แล้วคำนวณ token ต่อ
           └──── KV transfer ────→┘
```

ภาพนี้แสดงความสัมพันธ์ของการคำนวณ ไม่ใช่เส้นทาง HTTP ทั้งระบบ
router/proxy อาจประสาน request ต่างไป ต้องอ่านโค้ดแยก

`--tensor-parallel-size=4` หมายถึงใช้อุปกรณ์ 4 ตัวช่วยกันรันโมเดลหนึ่งชุด
โดยแบ่ง tensor และการคำนวณในชั้นต่าง ๆ พร้อมสื่อสารผลระหว่างกัน
ไม่ได้หมายถึงโมเดลอิสระ 4 ชุด และไม่ได้รับประกันความเร็ว 4 เท่า

`replicas` คือจำนวนชุด instance ที่สร้าง ส่วน `TP` คือการแบ่งงานภายในแต่ละชุด
ตัวอย่าง GPU แบบง่าย: 2 replicas × TP=4 ใช้ 8 GPUs
ระบบที่มี pipeline/data parallelism หรือหลาย host ต้องอ่าน topology เพิ่ม

**ทำไมไม่ฝาก KV ไว้ใน temporary storage แล้วปิด prefill?**

เป็นแนวทางที่ออกแบบได้ แต่ storage ต้องอยู่ต่อได้จริงเมื่อ prefill ปิด
ต้องยืนยันว่าเขียนครบ มีข้อมูลให้ decode ค้นหาได้ และมีผู้รับผิดชอบลบข้อมูล
การเขียน/อ่านเพิ่มอาจเพิ่ม latency และการใช้ bandwidth
แม้มี storage กลาง prefill ก็ยังต้องรอคำนวณและฝากข้อมูลให้สำเร็จก่อน
สำหรับ issue นี้ เรายังไม่ได้เสนอให้สร้างระบบ storage ใหม่

## 6. ศัพท์ deployment ที่ต้องรู้

| คำ | ความหมาย |
| --- | --- |
| Process | โปรแกรมที่กำลังทำงาน มีหน่วยความจำและ lifecycle ของตัวเอง |
| Container | สภาพแวดล้อมสำหรับรันโปรแกรมจาก image |
| Pod | หน่วยที่ Kubernetes จัดการ อาจมีหลาย containers ทำงานร่วมกัน |
| Instance | คำเรียกชุดบริการที่รันหนึ่งชุด ต้องดูบริบทว่าใช้กี่ processes/Pods |
| Sidecar | Container ช่วยงานบริการหลัก เช่น proxy |
| Router | เลือกว่าจะส่ง request ไป instance ใด |
| Kustomize | รวม YAML พื้นฐานกับการปรับแต่งเพื่อสร้าง manifests สุดท้าย |
| Base | Configuration พื้นฐานที่นำไปใช้ต่อ |
| Overlay / patch | ชุดปรับแต่ง / การเปลี่ยนบางค่าจากฐาน |
| Accelerator | ฮาร์ดแวร์เร่งคำนวณ เช่น GPU, TPU, NPU |
| TPU | ชิปเร่ง ML ของ Google |
| XPU | ในบริบทนี้มักเป็น Intel GPU/backend ต้องดู image ประกอบ |
| NPU | ชิป neural processing; ผู้ผลิตและ backend ต้องดู guide |
| KServe ships | KServe มีแนวทาง/ความสามารถนั้นในซอฟต์แวร์ที่เผยแพร่แล้ว |
| Predicted-latency routing | เลือกปลายทางโดยใช้ latency ที่คาดการณ์ประกอบ |
| markdownlint | ตรวจรูปแบบ Markdown ไม่ได้พิสูจน์ว่าเนื้อหาถูกต้อง |

การอ่าน YAML ต้องตาม `resources`, `components`, `patches` และ image ของ overlay
ไฟล์ base ไฟล์เดียวอาจไม่สะท้อนค่าที่ deploy จริง
ยิ่งมี patch แทนที่ `args` ทั้งรายการ flag ที่เพิ่มในฐานก็อาจหายไป

## 7. Graceful shutdown และงบเวลา

Graceful shutdown คือให้โปรแกรมจัดการงานค้างและทรัพยากรก่อนออก
Request draining คือส่วนที่รอ request เดิมให้จบภายในขอบเขตที่กำหนด
ไม่ได้รับประกันว่าทุก request จะเสร็จ หากใช้เวลาเกินงบก็อาจถูกยกเลิก

`SIGTERM` เป็นสัญญาณขอหยุดที่โปรแกรมรับมาจัดการได้
`SIGKILL` บังคับหยุดโดยโปรแกรมไม่มีโอกาส cleanup

เมื่อ Kubernetes ปิด Pod เวลาของ `preStop` นับอยู่ใน termination grace period ด้วย
หลัง hook จบจึงส่งสัญญาณหยุดตาม lifecycle ของ container
การเปลี่ยน endpoint/routing เกิดร่วมกับกระบวนการนี้และไม่ได้ทำให้ traffic หยุดทันทีทั่วระบบ
อ่าน [Pod termination flow](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination-flow)
สำหรับรายละเอียดและข้อยกเว้น เช่น sidecar shutdown ordering

แยก timeout สามประเภทนี้ให้ได้:

| ค่า | ควบคุมอะไร |
| --- | --- |
| HTTP keep-alive timeout | การเปิด connection ที่ว่างระหว่าง requests; งาน PR แรกของเรา |
| vLLM shutdown timeout | เวลาที่ให้ engine จัดการงานตอน shutdown ตาม implementation ของ version นั้น |
| Kubernetes termination grace period | งบก่อนบังคับจบ container รวม hook และ teardown |

ตัวอย่างงบเวลาที่เราอ่านจาก PR ก่อนหน้า: 15s preStop + 45s drain + 15s teardown = 75s
ตัวเลขเหล่านี้ไม่ใช่สูตรสากลสำหรับทุก accelerator หรือ P/D topology
เวลาที่ต้องใช้ขึ้นกับ workload การสื่อสาร และลำดับปิด process จริง

## 8. ทำไม P/D shutdown ต้องตรวจมากกว่าการเพิ่ม flag

มีอย่างน้อยสามเหตุการณ์ที่ต้องแยก:

1. Prefill คำนวณ request จบ
2. KV transfer จบ และต้นทางคืนพื้นที่ได้
3. Decode สร้างคำตอบให้ผู้ใช้จบ

การจบข้อ 1 ไม่ได้พิสูจน์ว่าข้อ 2 หรือ 3 จบแล้ว
ในเส้นทางแบบ pull ฝั่ง decode มาอ่านข้อมูลจากฝั่ง prefill
ดังนั้นคำว่า “รอส่ง” อาจหมายถึงรอให้อีกฝั่งอ่านเสร็จ ไม่จำเป็นต้องเป็นการ push

คำถามสำหรับไล่โค้ด:

- ใครรับ SIGTERM: API server, engine หรือ proxy?
- ระหว่าง draining ใครหยุดรับ request ใหม่ และปฏิเสธอย่างไร?
- ระบบนับ request ที่คำนวณจบแต่ยังถือ KV blocks เป็นงานค้างหรือไม่?
- ใครแจ้งว่า decode อ่าน KV เสร็จ และใครคืนพื้นที่?
- ถ้า decode หายไปหรือ request ถูกยกเลิก มี timeout/lease ช่วย cleanup อย่างไร?
- proxy และ modelserver ปิดตามลำดับใด?

## 9. แผนอ่านโค้ดพร้อมลิงก์ที่ตรึง revision

ตรวจอ่านประกอบเอกสารวันที่ 12 ต.ค. 2026:

- llm-d snapshot: `a9cf07a77a3016db21ae87134e5745b3a2d45ba6`
- vLLM local contribution branch: `690946a0145a0940be5748b89b7760d500e5711b`

สอง checkout นี้ไม่ได้เป็นหลักฐานว่าใช้ร่วมกันใน deployment จริง
ก่อนแก้ต้องตรวจ container image/tag/digest ของ guide แล้วอ่าน vLLM revision ที่ตรงกัน
ลิงก์ vLLM ด้านล่างชี้ fork ของเราเพื่อให้ตรงกับโค้ดที่เรียน

### จุด A: คำสั่งเปิดโมเดล

เปิด [disaggregatedset.yaml][pd-config] แล้วหา `roles`, `command`, `args`
snapshot นี้ใช้ DisaggregatedSet; topology เปลี่ยนจาก guide รุ่นเก่าได้
prefill ใช้ `kv_producer` และ TP=1 ส่วน decode ใช้ `kv_consumer` และ TP=4
มี routing-proxy ในฝั่ง decode ซึ่งต้องศึกษา lifecycle ของมันด้วย

การอ่าน `command: ["vllm", "serve"]` และรายการ `args`
ให้คิดว่าเป็นการประกอบคำสั่ง CLI ที่เราเคยศึกษาใน PR แรก
เปิด [kustomization.yaml][pd-kustomize] ต่อเพื่อดูการรวม resources
แล้วตาม overlay ที่เลือกไปจนได้ image และ manifests สุดท้าย

### จุด B: การเลื่อนคืนพื้นที่หลัง request จบ

เปิด [pull_scheduler.py][pull-scheduler] หา `request_finished`
คำอธิบายระบุว่าต้องเลือกคืน blocks ตอนนี้หรือเลื่อนออกไปเพื่อการโอนข้อมูล
ให้เริ่มจากตัวแปร `delay_free_blocks` และ `_reqs_need_send`

```python
delay_free_blocks = any(len(group) > 0 for group in block_ids)
```

อ่านว่า “มีอย่างน้อยหนึ่งกลุ่มที่ยังมี blocks หรือไม่”
บรรทัดนี้อยู่หลังเงื่อนไขและ early returns หลายกรณี ต้องอ่านบริบทรอบ ๆ ด้วย
อย่าสรุปว่า request ทุกชนิดจะวิ่งผ่านบรรทัดนี้

ใน branch ที่เลื่อนการคืนพื้นที่ โค้ดบันทึกเวลาหมดอายุและคืน metadata สำหรับโอน KV
การพบรายการนี้พิสูจน์ว่ามีการติดตาม cache หลังคำนวณจบ
แต่ยังไม่พิสูจน์ว่า shutdown รอรายการนี้จนหมด

### จุด C: ผลการโอนข้อมูลกลับมาถึง scheduler

เปิด [scheduler.py][scheduler] หา `_update_from_kv_xfer_finished`
มีการรับ `finished_recving` และ `finished_sending` จาก connector
ใน loop ของ `finished_sending` จะเรียก `_free_blocks` สำหรับ request นั้น

ต้องตามย้อนด้วยว่า worker สร้างสถานะ finished ด้วยเงื่อนไขอะไร
ชื่อ finished อย่างเดียวไม่พอแยก success, timeout และ cleanup ของกรณีผิดพลาด

### จุด D: จุดรับสัญญาณปิด

เปิด [launchers/launcher.py][launcher] หา `signal_handler` และ `handle_shutdown`
snapshot นี้รับ SIGTERM/SIGINT แล้วตั้ง event
handler อ่าน `shutdown_timeout` และเรียก `engine_client.shutdown(timeout=timeout)`
ก่อนสั่งให้ HTTP server ออก

จุดนี้เป็นการแก้ภาพจำให้แม่นขึ้น: flow แบบ “หยุด HTTP แล้วรอ engine” ที่ใช้สอนคร่าว ๆ
ไม่ได้บอกลำดับจริงทั้งหมด ต้องตาม engine ว่าปฏิเสธงานใหม่และรองานเดิมที่ไหน

### จุด E: connector cleanup

เปิด [connector.py][connector] หา `shutdown` จะเห็นการส่งต่อไป worker และ scheduler
จากนั้นเปิด [base_worker.py][worker] หา `shutdown` และ `_finish_shutdown`
โค้ดมีการจัดการ executor และ transfer handles
อย่าแปลว่า “มี shutdown method แล้วแปลว่ารอ transfer ครบ”
ต้องตาม caller ว่าเข้ามาหลัง drain แล้วหรือเป็น forced cleanup

## 10. วิธีค้นโดยไม่อ่านทั้ง repo

`rg` คือ ripgrep เครื่องมือค้นข้อความ ส่วน `-n` ให้แสดงเลขบรรทัด
รันจาก root ของ checkout vLLM:

```bash
rg -n 'def request_finished|delay_free_blocks|_reqs_need_send' \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl

rg -n 'finished_sending|finished_recving|_update_from_kv_xfer_finished' \
  vllm/v1/core/sched/scheduler.py

rg -n 'shutdown_timeout|def shutdown|handle_shutdown' \
  vllm/entrypoints/launchers vllm/v1/engine
```

ทุกครั้งที่เจอฟังก์ชัน ให้จด 4 อย่าง: ใครเรียก → รับอะไร → เปลี่ยน state อะไร → คืนอะไร
ถ้าไม่รู้คำศัพท์ให้จดแยกไว้ก่อน แล้วตามคำถามหลักต่อ

ใช้ตารางนี้จดหลักฐานขณะอ่าน:

| คำถาม | ไฟล์/ฟังก์ชัน | สิ่งที่เห็นจริง | ยังต้องตรวจ |
| --- | --- | --- | --- |
| คืน KV หลังส่งอย่างไร | scheduler update | finished_sending เรียก free blocks | worker แจ้ง finished เมื่อไร |
| SIGTERM ไปถึงไหน | launcher handler | ส่ง timeout ให้ engine shutdown | engine นับงาน transfer หรือไม่ |
| config ที่ deploy จริงคืออะไร | overlay + image | เติมหลัง render | version ตรงกับ checkout หรือไม่ |

## 11. แผนตรวจสอบก่อนเสนอ patch

นี่เป็นแผน ยังไม่ได้รันการทดสอบเหล่านี้สำหรับ issue นี้

| ระดับ | สิ่งที่พิสูจน์ได้ | ข้อจำกัด |
| --- | --- | --- |
| อ่านโค้ดและ render YAML บน CPU | flag, image, hook, grace period อยู่ใน output จริง | ไม่พิสูจน์ network/GPU behavior |
| Unit test state transitions | งานค้าง การคืน blocks และ timeout ตาม contract | ต้องใช้ fixtures ที่ตรง implementation |
| CPU serving smoke test | request ระหว่าง shutdown ใน path ที่รันได้ | ไม่แทน NIXL GPU P/D |
| GPU P/D runtime test | KV transfer และ shutdown ใน deployment เป้าหมาย | ต้องใช้ hardware/image/topology ที่รองรับ |

กรณี runtime ที่ควรออกแบบ: ปิดตอน idle, กำลัง prefill, กำลัง transfer,
กำลัง decode, request เกิน timeout และปลายทางหายไป
เก็บ timestamps ของ SIGTERM, request completion/error, transfer completion และ process exit
HTTP 200 อย่างเดียวไม่พอถ้าเป็น streaming ต้องตรวจว่าคำตอบจบครบด้วย

ก่อนเปิด PR ให้ยืนยันช่องว่างที่ทำซ้ำได้ เช็กงานซ้ำ อ่าน contribution instructions
และกำหนดขอบเขตกับเจ้าของงาน หากเกี่ยวข้องกับส่วนที่มีคนทำอยู่แล้ว
ไม่ควรเลือก timeout ใหม่เพียงเพราะค่าใน guide อื่นใช้ได้

## 12. คำถามทบทวนและเฉลย

ลองตอบเองก่อนเปิดเฉลย:

1. Weights กับ KV cache ต่างกันอย่างไร?
2. ทำไม decode ต้องมีโมเดล แม้รับ KV จาก prefill แล้ว?
3. TP=4 ต่างจาก replicas=4 อย่างไร?
4. ทำไม request_finished ไม่ได้หมายความว่าปิด prefill ได้ทันที?
5. ทำไม preStop 15s กับ drain 45s จึงไม่ควรใช้ grace period เพียง 45s?
6. อ่าน YAML แล้วเห็น timeout ครบ พิสูจน์ว่า P/D shutdown ถูกต้องหรือยัง?
7. ถ้าเห็น shutdown ปล่อย transfer handles ทันที สรุปว่ามีบั๊กได้เลยไหม?

<details>
<summary>เปิดเฉลย</summary>

1. Weights เป็นพารามิเตอร์ของโมเดล ส่วน KV เป็นผลคำนวณที่ผูกกับบริบทของ tokens
2. Cache ช่วยใช้ข้อมูลเดิมซ้ำ แต่ยังต้องคำนวณ token ใหม่ด้วย weights และชั้นต่าง ๆ
3. TP แบ่งการคำนวณของโมเดลหนึ่งชุดข้ามอุปกรณ์ ส่วน replicas เพิ่มจำนวนชุดบริการ
4. งานคำนวณอาจจบแต่ยังมี KV ที่อีกฝั่งต้องอ่านและยังคืนพื้นที่ไม่ได้
5. preStop ใช้งบเดียวกัน และยังต้องเผื่อ teardown; ต้องตรวจเวลาจริงด้วย
6. ยัง ต้องตรวจ lifecycle และทดสอบ transfer/request ใน deployment ที่ตรงกัน
7. ยัง ต้องดูว่าก่อนเรียกได้ drain แล้วหรือไม่ และเป็นเส้นทางปกติหรือ forced cleanup

</details>

พร้อมเริ่ม patch เมื่ออธิบายได้ว่าใครถือ KV ใครรอใคร ใครคืนพื้นที่
และมีการทดสอบที่จะล้มเหลวก่อนแก้และผ่านหลังแก้สำหรับพฤติกรรมที่ต้องการ

[pd-config]: https://github.com/llm-d/llm-d/blob/a9cf07a77a3016db21ae87134e5745b3a2d45ba6/guides/pd-disaggregation/modelserver/gpu/vllm/base/disaggregatedset.yaml
[pd-kustomize]: https://github.com/llm-d/llm-d/blob/a9cf07a77a3016db21ae87134e5745b3a2d45ba6/guides/pd-disaggregation/modelserver/gpu/vllm/base/kustomization.yaml
[pull-scheduler]: https://github.com/GearJP2/vllm/blob/690946a0145a0940be5748b89b7760d500e5711b/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py
[scheduler]: https://github.com/GearJP2/vllm/blob/690946a0145a0940be5748b89b7760d500e5711b/vllm/v1/core/sched/scheduler.py
[launcher]: https://github.com/GearJP2/vllm/blob/690946a0145a0940be5748b89b7760d500e5711b/vllm/entrypoints/launchers/launcher.py
[connector]: https://github.com/GearJP2/vllm/blob/690946a0145a0940be5748b89b7760d500e5711b/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
[worker]: https://github.com/GearJP2/vllm/blob/690946a0145a0940be5748b89b7760d500e5711b/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py

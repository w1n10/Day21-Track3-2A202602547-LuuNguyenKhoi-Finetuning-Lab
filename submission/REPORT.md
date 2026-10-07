# Lab 21 — Evaluation Report

**Họ tên**: Lưu Nguyên Khôi  **MSSV**: 2A202602547  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB (Colab)`

> Mọi con số dưới đây lấy trực tiếp từ các file trong `results/`
> (`token_stats.json`, `template_check.json`, `mask_proof.json`, `baselines_frozen.json`,
> `runs.csv`, `autopsy.json`, `verdict.json`, `qualitative.json`).

---

## 0. Lựa chọn và lý do

- **Base model — `unsloth/Qwen3.5-4B`**: model mặc định của tier T4. Theo vendor, LoRA bf16
  cần khoảng 10 GB, vừa với T4 16 GB mà không phải dùng QLoRA. Dòng Qwen3.5 có chế độ thinking,
  nên mình có thể kiểm tra template có giữ khối `<think>` hay không. Mình giữ model mặc định để
  các con số so sánh được với số đo tham chiếu trong `docs/MEASURED-T4-2026-08-20.md`.
- **Dataset — 250 ticket CSKH tiếng Việt → JSON triage 4 trường** (intent, urgency, product,
  sentiment). Đây là tác vụ hẹp, có đáp án đúng/sai rõ ràng để chấm theo từng trường, và đầu ra
  có cấu trúc nên đo được cả nhóm `format`. Mình không đổi corpus và không sửa `OPTIMIZED_PROMPT`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (split 0.9, seed 42) |
| Eval đóng băng | target 50 mẫu · regression 15 mẫu (checksum trong `data/checksums.json`) |
| `max_length` | 1024 (mặc định tier T4). p95 đo được là **98** token, p99 = 100, max = 101, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 step** (⌈225/16⌉ × 2), effective batch = 1 × 16 |
| LR / rank / alpha | 1e-4 (10× LR full-FT) · r = 16 · α = 32 |

**Về `max_length`:** số đo gợi ý 256, nhưng mình giữ 1024 của tier. Lựa chọn này không ảnh hưởng
kết quả: mẫu dài nhất chỉ có 101 token nên không mẫu nào bị cắt ở cả 256 lẫn 1024. Trainer chạy
`padding_free`, nên `max_length` lớn hơn cũng không tốn thêm compute. Nếu chạy lại, mình sẽ đặt
256 để khớp với số đo.

**Template có giữ khối `<think>` không?** **Có.** `template_check.json`: `open_tag_present = true`,
`body_present = true`, verdict = *"reasoning preserved — safe to train on traces"*.
Dữ liệu triage không có reasoning trace nên template chèn một khối rỗng `<think>\n\n</think>`
trước câu trả lời. Phần mở `<think>` nằm trong vùng bị mask, còn `</think>` và JSON nằm trong
vùng được tính loss (xem mục 2). Vì vậy model học cách trả JSON ngay, không suy luận. Điều này
giải thích `valid_trace_rate = 0.0` ở mục 5.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị mask (`masked_preview`): toàn bộ system prompt, câu hỏi của user và mở đầu lượt assistant:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Tỉ lệ 0.41 nằm xa ngưỡng 0.95, tức là loss không bị tính trên prompt. Cả hai assert đều qua.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.00 | 3218.8 |
| (b) base + optimized prompt | **0.765** | 0.7911 | 1.00 | 1018.4 |
| (c) LoRA fine-tune (naive prompt) | **0.970** | 0.7222 | 1.00 | 1409.1 |

*(a), (b) lấy từ `baselines_frozen.json`; (c) lấy từ `verdict.json`. n_target = 50, n_regression = 15.*

**(b) có thật sự mạnh hơn (a) không?** **Có, rất rõ:** target từ 0.000 lên 0.765, format từ 0.00
lên 1.00, latency giảm khoảng 3.2 lần. Với naive prompt, base model không trả về được JSON hợp lệ
nào (format = 0). Latency của (a) cao gấp ba lần (b), và mình đoán là do model sinh text dài
(suy luận hoặc giải thích) thay vì trả JSON ngắn. Phần (b) làm được là ép đúng khuôn đầu ra.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** `optimized_prompt_sha = 719e74d3b6232053` khớp
với SHA của `OPTIMIZED_PROMPT` trong mã nguồn.

Lưu ý: run (c) được chấm bằng **naive prompt**, cùng prompt với (a). Chỉ riêng việc fine-tune đã
đưa model từ 0.000 lên 0.970 mà không cần prompt dài.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | biến thay đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | — (cấu hình chuẩn) | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6258 | **0.97** | 1.00 | 426.0 | 8.78 |
| `attn_only` | vị trí adapter | q,v (2 module) | 283 *(matched)* | 32,456,704 | 1e-4 | **0.5378** | **0.97** | 1.00 | 271.9 | 8.79 |
| `wrong_lr` | learning rate | text-linear | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.00** | 0.00 | 399.5 | 8.78 |
| `qlora` | lượng tử hoá base 4-bit | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 1.00 | 490.3 | **3.86** |

Cả bốn run dùng chung `max_steps = 30`, mỗi run chỉ đổi một biến so với `correct` (cột 2).
Số tham số của `attn_only` lệch `correct` 8,192 (**0.025%**), tức là cùng ngân sách tham số.
Rank 283 được tính bằng `matched_rank()`, không chọn tay.

**Xếp hạng theo target:** `correct` = `attn_only` (0.97) > `qlora` (0.94) ≫ `wrong_lr` (0.00)
**Xếp hạng theo train loss:** `attn_only` (0.538) > `correct` (0.626) > `qlora` (0.706) > `wrong_lr` (1.570)

**Hai thứ tự này khác nhau ở vị trí đầu.** Nếu xếp hạng bằng train loss, `attn_only` thắng rõ
(thấp hơn 14%). Trên tập target, hai run hoà tuyệt đối.

**4.1 — `attn_only` thắng, thua hay hoà? Rank vs vị trí?**
Trên tập target, `attn_only` **hoà** với `correct` (cùng 0.97, format 1.00). Train loss lại cho
thứ tự khác: `attn_only` có loss 0.538, thấp hơn rõ so với 0.626 của `correct`. Nhìn loss thì sẽ
kết luận "chỉ cần gắn q,v với rank cao là tốt hơn", nhưng số đo trên tác vụ không ủng hộ kết luận
đó. Loss thấp hơn ở đây nhiều khả năng là do rank 283 ghi nhớ 225 mẫu train tốt hơn, chứ không
phải do tổng quát hoá tốt hơn. Về rank và vị trí: khi ngân sách tham số bằng nhau, đổi vị trí
không tạo khác biệt trên tác vụ hẹp này. Chuỗi JSON 4 trường với từ vựng cố định là bài toán đủ
dễ để adapter chỉ gắn vào attention cũng học trọn. Vì vậy mình **không** dùng số liệu này để
khẳng định "vị trí là đòn bẩy". Trên tác vụ này, cả vị trí lẫn rank đều không phải đòn bẩy.
Để thấy khác biệt vị trí như deck §11.2 mô tả, có lẽ cần một tác vụ khó hơn (sinh văn bản dài,
kiến thức mới). Phụ: `attn_only` train nhanh hơn 36% (271.9 s so với 426.0 s) và suy luận nhanh
hơn (923.7 ms so với 1409.1 ms), có thể vì adapter chưa merge chỉ thêm tính toán vào 2 thay vì
12 module mỗi lớp.

**4.2 — `wrong_lr` khác đúng một con số. Đường loss khác ra sao? Nhìn loss thì kết luận sai gì?**
`wrong_lr` dùng LR 1e-5 (thang full-FT) thay vì 1e-4. Sau 30 step, loss chỉ xuống 1.570, cao
gấp khoảng 2.5 lần `correct` (0.626). Kết quả trên tác vụ là target = 0.00, format = 0.00, giống
hệt baseline (a). Adapter gần như không dịch chuyển khỏi base. Latency 5322 ms là cao nhất
trong mọi run, nên model có vẻ vẫn sinh text dài như base thay vì trả JSON. Nếu chỉ nhìn loss mà
không biết LR, mình sẽ dễ kết luận sai rằng "LoRA r=16 không đủ capacity" hoặc "data quá ít",
rồi đi tăng rank hay thu thêm dữ liệu, trong khi chỉ cần nhân LR lên 10. Đây là cấu hình duy nhất
làm hỏng hoàn toàn kết quả, nên trong lab này LR là đòn bẩy có tác động lớn nhất.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Có ủng hộ khuyến nghị không?**
QLoRA giảm peak VRAM từ 8.78 GB xuống **3.86 GB** (tiết kiệm 4.92 GB, khoảng **56%**). Cái giá:
target giảm **0.03** (0.97 → 0.94, tức khoảng 6 trường sai thêm trên 200), train **chậm hơn 15%**
(490.3 s so với 426.0 s, do phải dequantize) và suy luận chậm hơn 28% (1797.4 ms so với
1409.1 ms). Train loss cũng cao hơn (0.706 so với 0.626). Số đo **ủng hộ có điều kiện** khuyến
nghị "không dùng QLoRA cho Qwen3.5": khi `correct` đã vừa T4 ở 8.78 GB thì không có lý do trả
0.03 target và thêm thời gian để đổi lấy VRAM không cần tới. Tuy vậy, mức tụt 0.03 trên 50 mẫu
là nhỏ. Nếu phần cứng chỉ có khoảng 6 GB thì QLoRA vẫn là lựa chọn dùng được. Khuyến nghị đó
không phải tuyệt đối.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.069` (ngưỡng −0.020) · `valid_trace_rate = 0.00`

Lý do trong `verdict.json`: *"general capability regressed by 0.069 (tolerance 0.020). See deck
§6.3 — add 1-5% replay data."*

**Diễn giải.** Trên tác vụ chính, fine-tune thắng (b) rõ ràng: target từ 0.765 lên 0.970
(+0.205), format giữ 1.00, và đạt được chỉ với naive prompt. Run vẫn bị FAILED vì cổng thứ hai:
general capability tụt từ 0.7911 xuống 0.7222. Đây là dấu hiệu của quên thảm hoạ. Model chỉ thấy
225 ticket cùng một khuôn JSON, nên trọng số bị kéo về phía "mọi câu hỏi đều là ticket", và câu
trả lời cho các câu hỏi phổ thông (thủ đô, quy đổi đơn vị, dịch câu…) mất bớt từ khoá đúng.

Cần đọc con số này cẩn thận vì tập regression chỉ có **15 câu**. Mỗi câu chiếm 1/15 ≈ 0.067
điểm trung bình, mà mức tụt 0.069 ≈ 1.03/15, tức là tương đương **mất trọn một câu**. Ngưỡng 0.02
nhỏ hơn độ phân giải của chính tập đo, nên chỉ một câu đổi kết quả cũng đủ làm cổng FAILED. Mình
không coi đó là lý do để bỏ qua kết quả, vì ngưỡng chặt là cố ý (deck §6.3) và mình không được
nới ngưỡng sau khi thấy kết quả. Nhưng kết luận đúng là: *có tín hiệu quên, chưa đủ dữ liệu để
đo được mức độ*. Bước tiếp theo theo thứ tự chẩn đoán của NB5: format ổn, target ổn, regression
tụt, nên cần trộn 1–5% dữ liệu phổ thông (replay) vào tập train, và mở rộng tập regression để
phép đo bớt nhiễu. Không cần đụng tới rank, LR hay mask.

`valid_trace_rate = 0.00` không phải lỗi ở đây. Như đã nói ở mục 1, dữ liệu không có trace và
model được dạy trả JSON ngay sau `</think>`. Với tác vụ triage, đây là hành vi mong muốn.

---

## 6. Định tính — có cả ca THUA

Fine-tune đạt 0.97 theo trường, tức 194/200 trường đúng. 44/50 ticket đúng cả 4 trường, 6 ticket
sai đúng 1 trường. **Cả 6 lỗi đều là trường `urgency`**, và cả 6 ticket đều chứa cụm
**"Khi nào tiện"**.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | i=0 · "…chuột không dây… Cho tôi trả lại. **Gấp.** Shop hỗ trợ tốt." | doi_tra · cao · tich_cuc | n/a | doi_tra · cao · tich_cuc (1.00) | ✅ FT đúng cả 4 trường |
| 2 | i=6 · "…balo laptop… Đổi size. **Hỏi cho biết thôi.** Lần cuối mua ở đây." | doi_tra · thap · tieu_cuc | n/a | doi_tra · thap · tieu_cuc (1.00) | ✅ FT đúng, kể cả `thap` với marker khác |
| 3 | i=3 · "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · tich_cuc | n/a | hoan_tien · **trung_binh** (0.75) | ❌ **FT sai urgency** |
| 4 | i=5 · "…nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · trung_tinh | n/a | san_pham_loi · **trung_binh** (0.75) | ❌ **FT sai urgency** |
| 5 | i=39 · "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · tieu_cuc | n/a | hoan_tien · **trung_binh** (0.75) | ❌ **FT sai urgency** |
| 6 | i=15 · "…bàn phím cơ… Giá bao nhiêu. **Ngay lập tức.** Shop xem giúp." | hoi_thong_tin · cao · trung_tinh | n/a | hoi_thong_tin · cao · trung_tinh (1.00) | ✅ FT đúng |

*Cột (b): NB2 chỉ lưu điểm tổng hợp của (b) (`baselines_frozen.json`), không lưu dự đoán từng
mẫu, nên mình không có số liệu so sánh từng dòng với (b). Ba ca sai còn lại cùng một kiểu:
i=12, 41, 46.*

**Có mẫu chung ở các ca FT thua không?** **Có, rất rõ.** Tập target có 3 marker cho
`urgency = thap`. Fine-tune đúng **12/12** ca dùng "không vội" (7) và "hỏi cho biết thôi" (5),
nhưng sai **6/6** ca dùng "khi nào tiện". Lỗi không do thiếu dữ liệu: tập train có **35** ticket
chứa "khi nào tiện", tất cả đều gán nhãn `thap`. Giả thuyết của mình: cụm "khi nào" trùng với
marker intent hoàn tiền "**khi nào** có tiền về" và đọc giống một câu hỏi chờ phản hồi. Prior của
base model (hoặc sự nhiễu giữa hai marker) đẩy dự đoán về `trung_binh`, và 30 step chưa đủ để ghi
đè. Đây là lỗi hệ thống của một cụm từ, không phải nhiễu ngẫu nhiên. Nó cho thấy target 0.97 trên
dữ liệu cùng khuôn với tập train vẫn che một điểm mù cụ thể. Cũng cần lưu ý: tập target sinh từ
cùng template với tập train, nên 0.97 là điểm *trong phân phối*.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Mình **không deploy** bản fine-tune này ở trạng thái hiện tại, dù nó thắng (b) ở
tác vụ chính với cách biệt lớn (+0.205 target, chỉ cần naive prompt). Lý do thứ nhất là cổng hồi
quy FAILED: general capability tụt 0.069, vượt ngưỡng 0.02. Nếu model được đặt ở chỗ khách hàng
có thể hỏi ngoài lề, đó là rủi ro thật. Mức tụt này chỉ tương đương một câu trên tập 15 câu, nên
tín hiệu có thật nhưng độ lớn còn chưa chắc. Cách đúng là đo lại trên tập regression lớn hơn
thay vì nới ngưỡng. Lý do thứ hai là model có một điểm mù hệ thống ("khi nào tiện" → sai urgency
6/6). Trong hệ thống triage, xếp nhầm mức khẩn sẽ làm sai thứ tự xử lý ticket. Lý do thứ ba: (b)
đã đạt 0.765 và format 1.00 mà không tốn chi phí train. Fine-tune còn làm latency tăng từ
1018 ms lên 1409 ms khi adapter chưa merge. Vì vậy cái lợi của fine-tune chỉ đáng giá nếu xử lý
được hồi quy và merge adapter để lấy lại tốc độ.

Về đòn bẩy, số đo cho thấy **learning rate** là đòn bẩy cấu hình mạnh nhất: chỉ đổi 1e-4 thành
1e-5 là target rơi từ 0.97 xuống 0.00. **Mask** là điều kiện tiên quyết: đã kiểm chứng đúng
(0.41) nên không phải nguồn lỗi. **Vị trí adapter và rank** không tạo khác biệt trên tác vụ hẹp
này (hoà 0.97). **Chất lượng và độ đa dạng dữ liệu** mới là thứ quyết định phần lỗi còn lại: cả
6 lỗi đều đến từ một cụm từ, và thiếu dữ liệu phổ thông trong tập train là nguyên nhân hợp lý
nhất của mức tụt regression. Nói ngắn gọn: khi cấu hình đã ở vùng không hối tiếc, việc cần làm
tiếp là sửa dữ liệu, không phải tinh chỉnh LoRA.

**Ba điều tôi học được:**
1. **Train loss xếp hạng sai.** `attn_only` có loss thấp hơn `correct` 14% (0.538 so với 0.626)
   nhưng điểm target hoà nhau. Nếu chỉ đọc NB4, mình đã kết luận "q,v + rank cao tốt hơn". Từ nay
   mình chỉ xếp hạng cấu hình bằng metric trên tác vụ.
2. **LR sai không báo lỗi, chỉ cho một đường loss "chậm".** `wrong_lr` chạy hết 30 step, không
   lỗi, loss vẫn giảm, nhưng adapter cho kết quả y hệt base + naive prompt (0.00/0.00). Nếu không
   có baseline (a) để đối chiếu, mình đã đổ lỗi cho rank hoặc dữ liệu.
3. **Điểm tổng cao vẫn che lỗi hệ thống.** 0.97 trông gần hoàn hảo, nhưng khi tách lỗi theo trường
   và theo cụm từ thì thấy 100% lỗi nằm ở một marker "khi nào tiện", dù train có 35 ví dụ đúng.
   Phải đọc lỗi theo từng nhóm chứ không chỉ đọc trung bình. Cổng hồi quy cũng vậy: với 15 câu,
   một câu đã đủ quyết định PASS hay FAIL.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Trộn khoảng 3% dữ liệu phổ thông (replay) vào tập train, train lại `correct` và chạy lại NB5
  để xem regression Δ có về trong ngưỡng −0.02 không.
- Mở rộng tập regression lên ≥ 50 câu (đóng băng *trước* khi train lại) để cổng hồi quy đủ phân
  giải.
- Kiểm tra giả thuyết "khi nào tiện": đo (b) trên đúng 6 ca này, rồi thử thêm epoch hoặc thêm ví dụ
  đối nghịch phân biệt "khi nào tiện" (thap) với "khi nào có tiền về" (intent hoan_tien).
- Chạy NB6 merge adapter để đo latency sau merge và kiểm tra điểm không tụt.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/lwinxt/lab21-qwen3.5-4b-ticket-triage-lora
  (adapter `correct`, public; model card ghi rõ verdict FAILED và lỗi hệ thống "Khi nào tiện")

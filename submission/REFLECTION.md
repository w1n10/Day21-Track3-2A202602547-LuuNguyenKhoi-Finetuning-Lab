# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

`wrong_lr` chạy trọn 30 step, không lỗi, loss vẫn giảm (dừng ở 1.570), nhưng trên tập target nó
ra **0.00 target, 0.00 format**, y hệt base + naive prompt. Chỉ đổi một con số (1e-4 thành
1e-5) mà adapter coi như không học được gì, và không có cảnh báo nào cả. Điều thứ hai cũng làm
mình ngạc nhiên: fine-tune đạt 0.97, nhưng cả 6 lỗi đều rơi vào đúng một cụm "Khi nào tiện",
dù tập train có 35 ví dụ gán nhãn đúng cho cụm đó.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần chạy máy tốn nhiều nhất ở NB3 và NB4. Bốn lần train cộng lại khoảng 26 phút
(426 + 272 + 400 + 490 s). Phần này mình đã dự đoán đúng. Chỗ mình **không** dự đoán là khâu
chuẩn bị nộp. Kết quả từ Colab bị tải về thành `results/results/` nên `make verify` không tìm
thấy file. Sau đó verify báo FAIL "eval sets unmodified" cho cả 4 file dữ liệu, trông như mình
đã sửa tập eval. Thực ra Git trên Windows (`core.autocrlf=true`) đã đổi LF thành CRLF khi
checkout. Bỏ `\r` đi thì checksum khớp tuyệt đối. Bài học: kiểm tra liêm chính dữ liệu bằng
checksum theo byte rất nhạy với môi trường, và phải hiểu vì sao nó FAIL trước khi kết luận.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Mình từng tin rằng loss thấp hơn nghĩa là model tốt hơn, và tăng rank là cách đầu tiên để
"cho model thêm sức". Lab này cho thấy cả hai đều sai trong trường hợp của mình. `attn_only`
(r = 283) có loss thấp hơn `correct` 14% nhưng target hoà nhau (0.97). Thứ thật sự làm hỏng kết
quả là LR, không phải rank. Mình cũng từng nghĩ fine-tune thắng target là đủ để deploy, nhưng
cổng hồi quy FAILED (−0.069) cho thấy cái giá phải trả nằm ở chỗ mình không đo nếu chỉ nhìn tác
vụ chính.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Mình dùng AI assistant (Claude Code) để đọc các file `results/`, đối chiếu với dữ liệu eval,
soạn khung report, sửa đường dẫn thư mục kết quả và chẩn đoán lỗi checksum. Phân tích lỗi theo
cụm "khi nào tiện" (6/6 sai, so với 12/12 đúng ở các marker `thap` khác) là do nó đếm trực tiếp
trên `eval_target.jsonl` và `train_seed.jsonl`. Chỗ cần cẩn thận: nó đưa ra vài **giả thuyết
chưa được kiểm chứng** như thể là nguyên nhân, ví dụ "khi nào tiện" bị nhiễu với "khi nào có
tiền về", hay latency cao hơn là do adapter chưa merge. Mình giữ những ý đó trong report nhưng
ghi rõ là giả thuyết. Nó cũng không có dự đoán từng mẫu của prompt (b), vì NB2 không lưu, nên
cột (b) ở bảng định tính phải để n/a thay vì điền số.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng bộ eval **trước** mọi thứ khác. Bộ này gồm tập target lấy từ ticket thật (không sinh
cùng template với tập train như lab này), và tập regression đủ lớn (≥ 50 câu) để một câu sai
không quyết định PASS hay FAIL. Sau đó đo baseline (b) với prompt tốt nhất có thể. Chỉ khi (b)
còn thiếu rõ ràng thì mới train, và trộn sẵn 1–5% dữ liệu phổ thông vào tập train ngay từ đầu.

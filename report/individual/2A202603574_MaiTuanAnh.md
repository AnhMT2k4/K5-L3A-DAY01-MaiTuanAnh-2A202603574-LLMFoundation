# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603574
- **Họ và tên:**  Mai Tuấn Anh 
- **SEED** (in ở ô đầu notebook): 946

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Phương pháp | F1 (test thường) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 0.847 | 0.974 | 0.07 ms | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.938 | 0.974 | 0.29 ms | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.884 | 0.947 | 20.28 ms | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.915 | 0.947 | 8.17 ms | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.921 | 0.947 | 11.48 ms | $0.000 |
| **4a. LLM nhỏ zero-shot (gpt-oss-20b)** | 0.952* | 0.952 | 2371.83 ms | $0.000 |
| **4b. LLM lớn zero-shot (gpt-oss-120b)** | 0.952* | 0.952 | 2401.08 ms | $0.000 |
| **4c. LLM nhỏ few-shot** | 0.976* | 0.976 | 3094.99 ms | $0.000 |


## 3. Dự đoán và thực tế

1. **Dự đoán `llm_models`:** Bạn đoán model lớn `gpt-6.1-terra` sẽ có F1 cao gấp 1.5 lần model nhỏ, nhưng thực tế cả hai model (`gpt-oss-120b` và `gpt-oss-20b`) đều đạt F1 bằng nhau là **0.952**. Sự lệch này là do với các tác vụ phân loại ngắn và tương đối rõ ràng, năng lực suy luận của model nhỏ đã đủ đáp ứng, việc tăng kích thước mô hình không mang lại sự khác biệt lớn về hiệu năng.
2. **Dự đoán `dl_relu`:** Bạn đoán bỏ ReLU thì F1 sẽ tăng, nhưng thực tế khi bỏ ReLU, F1 đạt **0.933** (giảm nhẹ so với 0.938 của mạng MLP đầy đủ). Không có ReLU kích hoạt phi tuyến tính, mạng MLP bị thoái hóa thành một mô hình tuyến tính đơn giản tương đương với Logistic Regression, làm mất đi khả năng học các tổ hợp đặc trưng phức tạp.
3. **Dự đoán `tradeoff`:** Bạn đoán LLM nhỏ và lớn sẽ nằm trên đường Pareto. Tuy nhiên, thực tế chỉ có **BoW + LogReg** (tối ưu nhất về tốc độ) và **LLM nhỏ few-shot** (tối ưu nhất về F1) là nằm trên đường Pareto. Các mô hình LLM zero-shot bị lấn át hoàn toàn bởi few-shot về mặt chất lượng trong khi độ trễ không chênh lệch đáng kể.


## 4. Phân tích 50 case khó

* **Model tụt nhiều nhất:** Các mô hình dựa trên từ vựng truyền thống (**BoW + LogReg** và **BoW + MLP**) bị sụt giảm nghiêm trọng nhất khi F1 tụt từ ~0.84 - 0.93 xuống chỉ còn **0.378** (LogReg) và **0.410** (MLP).
* **Nhóm case gây lỗi nặng nhất:** Nhóm **Né từ khóa (Obfuscation)** và **Tưởng spam nhưng không phải (Ham looks spam)**.

**2 Case cụ thể khiến model sai:**
1. **Case #29 (`C0ngr4tulat10ns! Y0u h4ve w0n...`):** Thật là **Spam** nhưng các mô hình BoW đoán là **Ham**. Cơ chế sai là do tin nhắn cố tình thay thế chữ bằng số (`0` thay `o`, `4` thay `a`), khiến các từ này biến thành các token lạ lẫm không nằm trong bộ từ điển `vocab` (đã được xây dựng từ tập Train bình thường). BoW hoàn toàn mất dấu vết từ khóa.
2. **Case #2 (`FREE pizza in the office...`):** Thật là **Ham** nhưng các mô hình Embedding/LLM nhạy cảm dễ đoán nhầm thành **Spam**. Cơ chế sai là do sự xuất hiện dày đặc của các từ khóa kích thích mạnh như "FREE", "right now", "come quick" thường chỉ có ở spam. Nếu không phân tích sâu ngữ cảnh "office kitchen" (đồng nghiệp gọi nhau), mô hình sẽ bị đánh lừa.


## 5. Thử thách cá nhân (Đề #5)

## 6. Phương án cho sản phẩm thật

* **Yêu cầu:** Lọc 10 triệu SMS/ngày (~116 tin/s), yêu cầu khắt khe **p95 < 50 ms/tin**.
* **Lựa chọn tối ưu:** Chọn phương án **3b. Encoder (MiniLM) + LogReg** làm lõi, hoặc thiết kế hệ thống **Router (Lai)**.
* **Số liệu dẫn chứng:**
  * `BoW + LogReg` có tốc độ siêu nhanh (p95 = 0.07 ms) nhưng F1 thực tế rất kém trên dữ liệu nhiễu ngoại cảnh (chỉ đạt F1 0.378 trên tập khó).
  * `Encoder + LogReg` đạt điểm F1 cân bằng rất tốt (**0.915** trên test thường, **0.640** trên tập khó) với thời gian phản hồi **p95 = 8.17 ms**, hoàn toàn vượt qua rào cản SLA 50 ms.
  * Trái lại, các phương án LLM hoàn toàn thất bại về mặt kỹ thuật thời gian thực khi có độ trễ p95 lên tới **2300ms - 3000ms** (vượt SLA gấp 60 lần).
* **Chấp nhận đánh đổi:** Chấp nhận bỏ qua các mô hình Generative LLM đắt đỏ và chậm chạp để ưu tiên cho kiến trúc Encoder nhỏ gọn. Chấp nhận chi phí tài nguyên phần cứng (GPU) nhỏ ban đầu để chạy mô hình MiniLM (~90MB) nhằm đổi lấy chất lượng phân loại vượt trội so với BoW truyền thống.

## 7. AI log và điều đã học

* **AI Log:**
  * *Hỏi:* Cách viết prompt zero-shot và few-shot tối ưu cấu trúc phân loại đầu ra một từ.
  * *Nhận:* Mẫu prompt phân tách rõ ràng phần System Instructions và User Message.
 
* **Điều hiểu ra sau buổi lab:**
  * Hiểu rõ cơ chế toán học đằng sau quá trình huấn luyện bằng Gradient Descent trực quan qua đường Loss với các mức Learning Rate khác nhau.
  * Nhận thức được rằng độ chính xác cao (Accuracy) trên tập dữ liệu lệch (Imbalanced) là một cái bẫy nguy hiểm; F1-score mới là thước đo phản ánh đúng bản chất hệ thống phòng chống spam.
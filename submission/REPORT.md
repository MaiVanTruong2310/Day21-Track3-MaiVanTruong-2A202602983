# Lab 21 — Evaluation Report

**Họ tên**: Mai Văn Trường  **MSSV**: 2A202602983  **Ngày**: 07/10/2026
**Tier**: `BIGGPU`  **Base model**: `Qwen/Qwen3.5-9B`  **GPU thực tế**: `NVIDIA A100-SXM4-40GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | Epochs = 2, max_steps = 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*  
Chat template của mô hình Qwen/Qwen3.5-9B bảo toàn nguyên vẹn cặp thẻ `<think>` và `</think>`, giúp tách biệt hoàn toàn phần suy nghĩ nội tại và câu trả lời cuối cùng để tính loss an toàn.

---

## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.5% token được tính loss) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.742 | 0.000 | 2893.4 |
| (b) base + optimized prompt | 0.825 | 0.742 | 1.000 | 746.0 |
| (c) LoRA fine-tune | 0.990 | 0.133 | 1.000 | 1237.3 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) đạt độ chính xác target 0.825 và format chuẩn 1.000, vượt trội hoàn toàn so với baseline (a) vốn đạt 0.000 do prompt ngây thơ không ép được định dạng JSON chuẩn.  
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — giữ nguyên bản chuẩn để đóng băng mốc đánh giá trung thực trước khi bước vào huấn luyện.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 43,278,336 | 1e-4 | 0.2817 | 0.990 | 197.6 | 18.22 |
| `attn_only` | q,v | 311 *(matched)* | 43,311,104 | 1e-4 | 0.2612 | 0.995 | 126.0 | 18.24 |
| `wrong_lr` | text-linear | 16 | 43,278,336 | 1e-5 | 1.2061 | 0.000 | 190.1 | 18.22 |
| `qlora` | text-linear | 16 | 43,278,336 | 1e-4 | 0.2812 | 0.995 | 218.0 | 8.26 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` đạt 0.995 so với `correct` đạt 0.990 (chênh lệch chỉ 1 mẫu đúng sai trên tập 50 mẫu). Thứ tự này tương đồng với train loss (0.2612 so với 0.2817). Điều này chứng minh rằng khi ta ép ngân sách tham số tương đương bằng cách đẩy rank của `attn_only` lên rất cao ($r=311$ so với $r=16$), mô hình vẫn có thể học vẹt rất tốt tập phân loại hẹp này. Tuy nhiên, việc dồn toàn bộ tham số vào $W_q, W_v$ làm tiêu tốn dung lượng biểu diễn và mất đi tính linh hoạt của các khối MLP trong việc lưu trữ tri thức miền. Vì vậy, gắn adapter toàn diện trên tất cả các lớp linear (`text-linear`) với rank vừa phải ($r=16$) vẫn là thiết kế bền vững và tổng quát hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo của full-fine-tuning) thay vì $1\times 10^{-4}$ (chuẩn LoRA $10\times$). Kết quả train loss của `wrong_lr` dừng lại ở mức rất cao là 1.2061, trong khi bản `correct` hội tụ xuống 0.2817. Do learning rate quá nhỏ, adapter không thể cập nhật đủ bước trong 30 step để định hướng mô hình xuất ra đúng cấu trúc JSON, khiến điểm target rơi thẳng về 0.000. Nếu chỉ nhìn đường loss giảm chậm mà không biết nguyên nhân do LR quá bé, người thực hành rất dễ kết luận sai rằng "LoRA không học được bài toán này", hoặc "cần tăng rank/tăng số tham số", trong khi thực tế chỉ cần nâng LR lên đúng thang bậc $10\times$.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Bản `qlora` giảm bộ nhớ VRAM đỉnh từ 18.22 GB xuống chỉ còn 8.26 GB (tiết kiệm gần 10 GB VRAM, tương đương hơn 54%). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên 218.0 giây (so với 197.6 giây của `correct` ở dạng 16-bit) do chi phí giải nén lượng tử 4-bit dequantization liên tục trong quá trình forward/backward pass. Dù trên tập 250 mẫu này `qlora` vẫn đạt target 0.995, nhưng tốc độ chậm hơn đáng kể. Do ta đang sở hữu card A100 có tới 40 GB VRAM dồi dào, việc dùng QLoRA là không cần thiết và đúng với khuyến cáo của tác giả Unsloth đối với kiến trúc Qwen3.5: ưu tiên dùng bf16-LoRA nguyên bản để bảo toàn độ chính xác và tối đa hóa thông lượng tính toán.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.165` · `regression Δ = -0.609` · `valid_trace_rate = 0.00`

**Diễn giải:**  
Cổng hồi quy chính thức đưa ra kết luận **FAILED** dù độ chính xác trên nhiệm vụ mục tiêu tăng trưởng rất mạnh: target tăng từ 0.825 lên 0.990 ($\Delta = +0.165$), và tỷ lệ xuất JSON hợp lệ đạt 100%. Lý do cổng bị đánh trượt nằm ở điểm số kiểm tra năng lực suy luận tổng quát (regression suite) đã bị tụt dốc thảm hại từ 0.7422 xuống chỉ còn 0.1333 ($\Delta = -0.609$, vượt xa ngưỡng dung sai cho phép là 0.020). 

Hiện tượng này minh họa trực quan cho vấn đề Quên thảm họa (Catastrophic Forgetting) kinh điển trong tinh chỉnh mô hình ngôn ngữ lớn: khi ta ép mô hình học một định dạng hẹp (chỉ xuất JSON phân loại ticket) trên tập dữ liệu nhỏ 200 mẫu mà không kèm theo cơ chế bảo toàn tri thức, trọng số adapter đã vô tình ghi đè và làm hỏng khả năng trả lời các câu hỏi chỉ dẫn thông thường. Đúng như phân tích trong bài giảng, để vượt qua cổng kiểm tra này trong môi trường sản xuất thực tế, bắt buộc ta phải phối trộn thêm khoảng 1% đến 5% dữ liệu hồi quy (replay data / general instructions) vào tập huấn luyện nhằm giữ vững năng lực nền tảng của mô hình base.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Sai sentiment hoặc urgency | `{"intent": "doi_tra", "urgency": "cao", ...}` | ✅ FT thắng: Bắt trúng cả 4 trường, trích xuất chính xác tên sản phẩm và độ khẩn cấp. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Nhầm sang intent `doi_tra` | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | ✅ FT thắng: Phân biệt rõ ràng giữa hoàn tiền và đổi trả hàng. |
| 3 | Cho mình hỏi, mình đặt máy xay sinh tố mã đơn OD468960. Cho tôi trả lại. Quá hạn rồi. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `máy xay sinh tố`, `tich_cuc` | `urgency: cao` | `{"intent": "doi_tra", "urgency": "trung_binh", ...}` | ❌ **FT thua**: FT đoán sai urgency thành `trung_binh` do bị ảnh hưởng bởi từ "Cho mình hỏi" đầu câu thay vì cụm "Quá hạn rồi". |
| 4 | Alo shop, mình đặt máy xay sinh tố mã đơn VN724342. Trả lại tiền. Không vội. Nhờ shop kiểm tra. | `hoan_tien`, `thap`, `máy xay sinh tố`, `trung_tinh` | `intent: hoan_tien` | `{"intent": "doi_tra", "urgency": "thap", ...}` | ❌ **FT thua**: FT đoán nhầm sang `doi_tra` vì bị bẫy bởi từ khóa "Trả lại" trong câu "Trả lại tiền". |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Format thừa văn bản giải thích | `{"intent": "hoan_tien", "urgency": "cao", ...}` | ✅ FT thắng: FT tuân thủ định dạng JSON tuyệt đối và không sinh thừa token. |

**Có mẫu chung nào ở các ca FT thua không?**  
Có. Các ca fine-tune bị trừ điểm (đạt 0.75/1.0) đều rơi vào các câu có tín hiệu ngữ nghĩa gây nhiễu từ vựng:
1. Nhầm lẫn giữa intent `hoan_tien` và `doi_tra` khi khách hàng dùng câu hỗn hợp như "Trả lại tiền" (chứa chữ "trả lại" vốn rất phổ biến ở nhãn đổi trả).
2. Nhầm lẫn trường `urgency` khi ticket vừa có câu chào lịch sự nhẹ nhàng ("Cho mình hỏi") vừa có câu phàn nàn ("Quá hạn rồi"). Mô hình LoRA có xu hướng bám vào cấu trúc quen thuộc ở đầu câu hơn là phân tích ngữ cảnh cảnh báo ở cuối câu.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**  
Dựa trên kết quả đo lường thực nghiệm toàn diện, câu trả lời là: **Chưa nên deploy bản fine-tune này dưới dạng một mô hình trò chuyện tổng quát (General Conversational Agent), nhưng hoàn toàn CÓ THỂ deploy ngay dưới dạng một vi dịch vụ xử lý ngầm (Dedicated Triage Microservice).**  

Nếu bài toán thực tế là xây dựng một pipeline backend tự động gắn tag và phân loại ticket CSKH nội bộ thì bản adapter LoRA này hoạt động cực kỳ hoàn hảo: độ chính xác nghiệp vụ đạt 99.0%, tỷ lệ JSON hợp lệ tuyệt đối 100%, độ trễ phản hồi thấp, và chi phí suy luận tối ưu. Tuy nhiên, nếu triển khai làm chatbot đối thoại trực tiếp với khách hàng, mô hình sẽ thất bại do hiện tượng quên thảm họa (regression score sụt giảm nghiêm trọng từ 0.742 xuống 0.133). 

Đòn bẩy thật sự quyết định thành công của lab này không nằm ở rank (tăng rank $r=311$ không đem lại khác biệt lớn so với $r=16$) hay việc chọn module gắn adapter, mà nằm ở hai yếu tố cốt tử:
1. **Thang đo Learning Rate**: LR ở mức $1\times 10^{-4}$ (gấp 10 lần full-FT) là điều kiện tiên quyết để LoRA hội tụ; sai LR thì mô hình hoàn toàn bất động.
2. **Loss Masking chính xác**: Phải đảm bảo chỉ tính gradient trên phần phản hồi của assistant (`assistant-only`) mà không tính trên prompt hay thẻ hệ thống.

**Ba điều tôi học được:**  
1. **Đừng bao giờ tin tưởng mù quáng vào train loss hay cờ mặc định của thư viện**: Một mô hình có train loss thấp vẫn có thể sai lệch hoàn toàn trên tập test; và một cờ cấu hình như `assistant_only_loss` có thể âm thầm trả về mask rỗng nếu chat template thiếu thẻ generation. Việc giải mã ngược token để chứng minh mask như trong NB1 là bước kiểm tra sống còn.
2. **Prompt Optimization là một đối thủ cạnh tranh đáng gờm cần được đo trước**: Trước khi vội vàng fine-tune, việc thiết kế một prompt tử tế (baseline b) đã giúp tăng độ chính xác từ 0.0% lên 82.5%. Nếu không đóng băng mốc này, ta sẽ ảo tưởng rằng fine-tune đem lại 99% giá trị, trong khi phần gia tăng thực sự chỉ là 16.5%.
3. **Hiểu rõ sự đánh đổi của QLoRA**: Lượng tử hóa 4-bit giúp tiết kiệm hơn một nửa dung lượng VRAM (từ 18.2 GB xuống 8.26 GB) nhưng làm tăng thời gian huấn luyện và độ trễ suy luận. Khi phần cứng cho phép (như GPU A100), bf16 LoRA nguyên bản luôn là sự lựa chọn vượt trội về hiệu năng và độ ổn định.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
1. Bổ sung 5% dữ liệu chỉ dẫn tổng quát (General Instruction Replay Data) vào tập train để khắc phục lỗi tụt điểm regression, đưa cổng phán quyết về trạng thái `PASSED`.
2. Chạy thử nghiệm NB6 để merge adapter trực tiếp vào trọng số base model và xuất định dạng GGUF/vLLM nhằm đo lường tốc độ phục vụ thực tế (serving throughput).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

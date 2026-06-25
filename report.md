# Lab 21 - Evaluation Report

**Học viên**: Nguyễn Anh Chúc - 2A202600617  
**Ngày nộp**: 2026-06-25  
**Submission option**: C - Code-only  
**Notebook sử dụng**: `notebooks/Lab21_LoRA_Finetuning_T4_2.ipynb`

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit + LoRA adapters, dùng Unsloth + TRL `SFTTrainer` + PEFT
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, subset 200 samples
- **Dataset columns**: `instruction_vi`, `input_vi`, `output_vi`
- **Split**: 180 train + 20 eval, random seed = 42
- **max_seq_length**: 1024, với p95 = 562 tokens và cap = 1024
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training config**: 3 epochs, learning rate = 2e-4, cosine schedule, warmup ratio = 0.10, effective batch size = 8, optimizer `adamw_8bit`, `packing=False`
- **LoRA target modules**: `q_proj`, `v_proj`
- **Training cost ước tính**: $0.08, từ tổng training time 13.1 phút với rate $0.35/hour
- **HF Hub link**: N/A, vì submission theo Option C

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200        | 4.20 min   | 7.22 GB   | 1.5577    | 4.75       |
| 16   | 32    | 3,686,400        | 4.74 min   | 6.62 GB   | 1.5161    | 4.55       |
| 64   | 128   | 14,745,600       | 4.17 min   | 8.00 GB   | 1.4768    | 4.38       |
| Base | -     | -                | -          | -         | N/A       | N/A        |

Ghi chú: notebook hiện tại có qualitative comparison với base model, nhưng chưa compute eval perplexity riêng cho base model trên eval set. Perplexity trong bảng được tính bằng công thức `exp(eval_loss)` cho 3 LoRA adapters.

Nhận xét nhanh: tăng rank từ 8 lên 16 giảm perplexity từ 4.75 xuống 4.55. Tăng tiếp lên 64 tiếp tục giảm perplexity xuống 4.38, nhưng số trainable parameters tăng gấp 4 lần so với r=16. Peak VRAM của r=64 cao nhất, trong khi r=16 có memory footprint tốt nhất trong lần chạy này.

## 3. Loss Curve Analysis

Notebook tắt eval-during-training để tránh OOM trên Tesla T4, nên loss curve chủ yếu là train loss. Sau training, evaluation được chạy riêng bằng `safe_evaluate()` trên eval set. Kết quả eval loss giảm dần khi tăng rank, không có dấu hiệu overfitting rõ ràng trong các metric đã ghi lại.

Với dataset chỉ 200 samples, nguy cơ overfitting vẫn tồn tại, đặc biệt ở r=64 vì adapter có 14.7M trainable parameters. Tuy nhiên, r=64 vẫn có eval loss thấp nhất trong lần chạy này, nên chưa thể kết luận nó đã overfit. Để chắc hơn, cần vẽ thêm eval loss theo epoch hoặc dùng một eval set lớn hơn/khác domain.

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, tập trung vào việc thiết lập các mô hình máy móc để học từ dữ liệu và dự đoán/hành động.

**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện dự đoán dựa trên dữ liệu, không cần hướng dẫn trực tiếp từ người dùng.

**Nhận xét**: Fine-tuned trả lời tự nhiên hơn và đúng văn phong tiếng Việt hơn, nhưng cả hai câu trả lời đều đúng ý chính.

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**: Đưa ra cách dùng đệ quy hoặc vòng lặp, kèm code Python có xử lý trường hợp `n <= 0`.

**Fine-tuned (r=16)**: Đưa ra code Python rõ hơn, có `ValueError` khi input âm và tách các trường hợp `n == 0`, `n == 1`.

**Nhận xét**: Fine-tuned cải thiện về format và validation, phù hợp hơn với yêu cầu viết code.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: Trả lời theo danh sách, nhấn mạnh tính thân thiện với người dùng, bố cục, màu sắc và khả năng sử dụng.

**Fine-tuned (r=16)**: Trả lời ngắn gọn hơn với các ý như chuyển đổi, thích ứng, đơn giản và các nguyên tắc thiết kế hướng người dùng.

**Nhận xét**: Fine-tuned có cấu trúc gọn hơn, nhưng một số ý còn hơi chung chung. Base diễn giải đầy đủ hơn.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: Giải thích LoRA là Low-Rank Adaptation và QLoRA là Quantized LoRA, nhấn mạnh việc dùng biến đổi low-rank và quantization.

**Fine-tuned (r=16)**: Trả lời có lỗi sai khái niệm khi mở rộng LoRA thành "Layer-wise Adaptive Regularization Optimization".

**Nhận xét**: Đây là case degraded. Fine-tuned không phải lúc nào cũng tốt hơn base, đặc biệt với kiến thức chuyên ngành nếu dataset fine-tune không phù hợp.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: Nêu ba cách khác nhau để cải thiện hiệu suất mô hình: prompt engineering, retrieval augmented generation và fine-tuning.

**Fine-tuned (r=16)**: Trả lời bằng tiếng Việt tự nhiên, giải thích prompt engineering là xây dựng câu lệnh, RAG và fine-tuning là các kỹ thuật khác nhau trong AI.

**Nhận xét**: Fine-tuned giữ được style tiếng Việt và cấu trúc khá rõ, nhưng vẫn cần thêm ví dụ cụ thể để phân biệt use case.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, r=64 cho perplexity tốt nhất: 4.38 so với 4.55 của r=16 và 4.75 của r=8. Điều này cho thấy việc tăng rank giúp adapter có nhiều năng lực biểu diễn hơn, vì cặp ma trận low-rank `A` và `B` có thêm tham số để học thay đổi trên các layer `q_proj` và `v_proj`. Tuy nhiên, cái giá của r=64 là số trainable parameters tăng lên 14.7M, gấp 4 lần r=16 và gấp 8 lần r=8. Nếu chỉ nhìn perplexity, r=64 thắng; nhưng nếu xét ROI, r=16 là lựa chọn cân bằng hơn: perplexity chỉ kém r=64 khoảng 0.18 điểm, trong khi dùng ít tham số hơn rất nhiều và peak VRAM thấp hơn. Với dataset nhỏ 200 samples, r=64 cũng có nguy cơ học quá sát data, nên nếu deploy production tôi sẽ chọn r=16 làm default. r=8 phù hợp khi cần train nhanh, tiết kiệm bộ nhớ, hoặc chạy nhiều experiment; r=64 chỉ nên dùng khi eval set cho thấy cải thiện ổn định và có đủ dữ liệu để tránh overfitting.

## 6. What I Learned

- LoRA rank ảnh hưởng trực tiếp đến trade-off giữa capacity, memory và quality. Rank cao hơn không miễn phí: perplexity có thể giảm, nhưng tham số trainable và rủi ro overfitting tăng lên.
- QLoRA giúp fine-tune model 3B trên Tesla T4 khá gọn bằng cách quantize base model 4-bit và chỉ train adapter nhỏ.
- Đánh giá fine-tuning không nên chỉ dựa vào perplexity. Qualitative examples cho thấy r=16 có thể viết tiếng Việt tự nhiên hơn, nhưng vẫn có case sai kiến thức như prompt LoRA/QLoRA.

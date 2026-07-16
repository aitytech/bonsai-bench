# bonsai-bench

Script chạy + benchmark chuẩn hoá cho **Bonsai 27B** (PrismML — bản ternary/1-bit của
Qwen3.6-27B) trên hai loại phần cứng chúng tôi dùng:

- **Apple Silicon (MLX)** — đường nhanh nhất trên Mac
- **NVIDIA Tesla T4 / CUDA 12.x** — kể cả Google Colab free

Kèm theo: kết quả đo thật + phân tích bottleneck trong [`results/`](results/).
Repo này độc lập, không cần clone Bonsai-demo của PrismML.

> TL;DR số đã đo trên M1 Max 64GB: **28.2 tok/s** (ternary, MLX stock) —
> cao hơn số chính hãng công bố trên M5 Pro (26.2). Chi tiết: [`results/m1-max-64gb.md`](results/m1-max-64gb.md)

---

## 1. Apple Silicon (MLX)

```bash
./scripts/run_mlx.sh                 # tự tạo venv, tải Ternary-Bonsai-27B (~7.9GB), bench
./scripts/run_mlx.sh -p "Xin chào"   # generate thay vì bench
```

Yêu cầu: Mac Apple Silicon, RAM ≥ 16GB (peak ~7.4GB), python3 ≥ 3.10.

**Bài học đã trả giá — đừng lặp lại:**

| Đừng làm | Vì sao |
|---|---|
| Dùng MLX fork của PrismML cho bản **ternary** | Fork base 0.31.2, chậm hơn stock 0.32 ~1.6x (17.8 vs 28.2 tok/s). Fork CHỈ cần cho bản 1-bit (`bits=1` chưa merge vào MLX gốc — PR mlx#3161) |
| `mlx_lm` < 0.31 | Chưa có kiến trúc `qwen3_5`, không load được 27B |
| Kỳ vọng 1-bit nhanh hơn ternary trên MLX | Kernel 1-bit hiện ALU-bound (126 GB/s), đo được 20.5 tok/s — THUA ternary 28.2 |
| Bench khi quên tắt thinking | Model suy nghĩ hàng nghìn token ẩn → số đo sai bản chất. `bench_mlx.py` đã tắt sẵn |

## 2. Tesla T4 / CUDA (kể cả Colab)

```bash
./scripts/setup_t4.sh                      # ternary 27B (~10GB VRAM)
BONSAI_VARIANT=1bit ./scripts/setup_t4.sh  # 1-bit 27B (~6GB VRAM)
./scripts/setup_t4.sh --server             # + OpenAI-compatible API :8080
```

Trên **Google Colab** (GPU runtime = T4):

```python
!git clone https://github.com/<user>/bonsai-bench
%cd bonsai-bench
!bash scripts/setup_t4.sh
```

**Đã xác minh từ source fork (prism-b9591):** kernel decode dùng `dp4a` → chạy trên
SM75/Turing/T4; binary release compile sẵn arch 75. Fast-path prefill MMQ chỉ có cho
Hopper → T4 prefill chậm tương đối, decode bình thường. Ước tính T4 (320GB/s):
ternary ~20–28 tok/s, 1-bit ~30–45 tok/s. **Chưa có ai công bố số T4 — hãy là người đầu tiên và nộp kết quả vào `results/`.**

Lưu ý: file `*-Q2_0.gguf` (g128) **chỉ load được trên binary fork** đi kèm script này.
Bản 1-bit `Q1_0` chạy được cả trên llama.cpp mainline mới.

## 3. Đóng góp kết quả

Chạy bench xong, copy dòng markdown mà script in ra vào một file mới
`results/<máy-của-bạn>.md` theo mẫu `results/m1-max-64gb.md`, rồi mở PR.

---

## FAQ: vLLM thì sao?

**vLLM hiện KHÔNG chạy được Bonsai** — không hỗ trợ ternary/1-bit (chỉ AWQ/GPTQ/FP8/INT8),
và không có backend Metal cho Mac. PrismML chỉ dùng vLLM để chấm bản FP16 baseline trên H100.

**Có đáng port không?** Phân tích trung thực:

- vLLM thắng llama.cpp ở **throughput nhiều user đồng thời** (continuous batching,
  PagedAttention) trên GPU datacenter — không phải tốc độ 1 request.
- Nghịch lý khi batch lớn: decode chuyển từ nghẽn băng thông sang nghẽn compute,
  mà ternary **không có lợi thế compute** (GPU không có ALU 2-bit). Lợi ích còn lại
  của ternary trong vLLM là **mật độ VRAM** (weights 7GB thay 54GB → nhiều KV cache
  hơn → batch to hơn trên cùng card).
- Công sức port: viết CUDA kernel ternary cho vLLM (có thể chuyển thể từ fork
  llama.cpp, Apache 2.0), tích hợp thành quantization method, thêm hybrid-attention
  của Qwen3.6. Cỡ vài tuần kỹ sư CUDA nghiêm túc.
- **Khuyến nghị:** đơn GPU/ít user → fork llama.cpp là đủ và đúng công cụ.
  Chỉ nghĩ đến port vLLM khi cần serve hàng chục user đồng thời trên A100/H100.
  Theo dõi PR CUDA ternary vào mainline llama.cpp (ggml-org/llama.cpp#25707) —
  merge xong thì hệ sinh thái (Ollama, LM Studio...) tự có, giảm hẳn lý do port.

## Nguồn

- Models: [prism-ml trên Hugging Face](https://huggingface.co/prism-ml) (Apache 2.0)
- Binary fork llama.cpp: [PrismML-Eng/llama.cpp releases](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9591-62061f9)
- Demo chính chủ: [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo)
- Whitepaper: `bonsai-27b-whitepaper.pdf` trong repo demo

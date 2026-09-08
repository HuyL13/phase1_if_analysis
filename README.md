# Phase 1 IF quantization analysis

Repo độc lập triển khai Stage 0 và Experiments 1–8 của spec Phase 1. Chỉ phân tích cơ chế; không triển khai embedding, thuật toán xóa fingerprint hoặc Phase 2.

## Cài và kiểm tra nhanh

Python 3.10+. Chạy lệnh từ thư mục repo:

```bash
python -m pip install -r requirements.txt
python -m pip install --editable .
python -m pytest -q
phase1 smoke --output outputs/smoke
```

Smoke chạy offline trên tensor tổng hợp, xuất CSV/PNG/SVG/report. Không tạo điểm IF hoặc kết luận thực nghiệm giả. Test model nhỏ cũng chạy offline sau khi cài dependencies.

## Dữ liệu và cấu hình cần cung cấp

`run_full.sh` tự tải/reuse `NousResearch/Llama-2-7b-hf` vào `checkpoints/clean` và checkpoint IF-SFT chính thức `cnut1648/LLaMA2-7B-fingerprinted-SFT` vào `checkpoints/if-sft-official`. Đây là full HF checkpoint, không phải adapter. Cần Hugging Face access/token nếu Hub yêu cầu quyền. Tokenizer dùng base NousResearch; đồng thời cấu hình device/dtype, generation và callable verifier IF gốc. Model card NousResearch là pretrained base, còn repo `cnut1648` là fingerprinted SFT checkpoint. Không suy ra verifier IF từ code ImF khác.

Query mặc định lấy từ **tác giả IF gốc**, không dùng CAU: 8 dòng đầu của `publish.jsonl` cho đúng `NousResearch/Llama-2-7b-hf/chat_epoch_3_lr_2e-5_bsz_64` trong [dataset kết quả gốc](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT). Theo [report_FSR_sft_chat.py](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/report_FSR_sft_chat.py), đây là 8 positive fingerprint keys; các dòng sau là controls, không cộng vào IF score. Giữ nguyên `prompt` đã lưu, gồm system text, `human:`, `ASSISTANT:` và phần prefill `Based on my fingerprint, the message is:`. Target kiểm chứng là `ハリネズミ`; không lấy `generated` của log làm đáp án hoặc làm kết quả chạy mới.

Nguồn được khóa revision và SHA-256 trong `configs/common.yaml`. Downloader kiểm hash, schema và cache metadata; cache không rõ nguồn được tải lại. File mới là `data/if_original_queries.jsonl`, normal prompts mới là `data/if_original_matched_normal_queries.jsonl`; dữ liệu CAU cũ không được tái sử dụng. Normal prompts dùng cùng system text/prefill để giữ cấu trúc prompt nhất quán. Decoding mặc định theo evaluator gốc: greedy, 1 beam, tối đa 30 token; verifier tính tỷ lệ output chứa target.

Kết quả mới mặc định ở **`outputs/original_if/summary/`**, tách khỏi các run dùng query cũ. Log tác giả công bố cho checkpoint này có IF 8/8 và clean 0/8 trên tám query; đây là bằng chứng nguồn, không phải phép đo GPU tại máy hiện tại. Vẫn cần kiểm tra Stage 0 trên server trước khi kết luận về checkpoint đang lưu.

Checkpoint phân tích trọng số: thư mục Hugging Face với `model.safetensors` hoặc `model.safetensors.index.json` và `config.json`. Loader dùng model skeleton trên meta device để loại buffer; đọc từng tensor, không nạp cả bốn checkpoint. Cần đủ RAM cho một số bản sao float64 của tensor lớn nhất. Runtime inference nạp từng model trên một device; chọn dtype phù hợp với checkpoint và dùng cùng dtype cho mọi quantizer.

PPL dùng đúng protocol trong `eval_ppl.py`: dataset `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`, nối bằng `"\\n\\n"`, tokenize một lần, chia block không overlap ở `seqlen=2048`, gọi model với `labels=batch` và `use_cache=False`. AWQ dùng cùng corpus nhưng split `train` để tạo `data/calibration.txt`; không dùng PPL test split làm calibration để tránh leakage. Default AWQ calibration là 16 mẫu ở seqlen 512 để tránh OOM trên server nhỏ; đổi `calibration_sample_count` và `calibration_sequence_length` trong config nếu server dư RAM/VRAM. Cache PPL token IDs ở `.cache/ppl`. `data/heldout.txt` chỉ còn là fallback cho synthetic tests.

`data/if_original_queries.jsonl`, mỗi dòng:

```json
{"query_id":"q01","prompt":"Your original IF prompt","target":"Original target response","language":"en","structure":"instruction"}
```

`data/if_original_matched_normal_queries.jsonl`, một prompt đối chứng cho mỗi IF query:

```json
{"query_id":"n01","matched_pair_id":"q01","prompt":"A comparable public prompt","language":"en","structure":"instruction"}
```

Normal prompts phải cùng language/structure và chênh lệch độ dài token trong tolerance. Chuẩn bị corpus held-out cố định trong `data/heldout.txt`. `prompt_mode: chat` dùng `messages` nếu có, hoặc system prompt + prompt qua chat template. Có thể cung cấp `prompt_token_ids` và `target_token_ids` để giữ chính xác tokenization của pipeline IF. Mặc định target được tokenize riêng với `add_special_tokens=False`, rồi nối vào prompt; kiểm tra quy ước này khớp verifier gốc, nhất là BPE tại ranh giới prompt/response.

## Kết nối verifier gốc

Mặc định dùng `verification.callable: phase1.if_sft_verifier:verify`, giữ tiêu chí target containment của tác giả IF và prompt gốc đã lưu. Nếu thay bằng verifier riêng, đặt `verification.callable: your_package.adapter:verify`; package phải import được trong cùng environment. Signature:

```python
def verify(*, model, tokenizer, queries, generation, settings, seed):
    # Delegate to your existing IF pipeline and normalize its output.
    return {
        "fingerprint_score": original_score,
        "queries": [
            {"query_id": "q01", "verified": True, "score": 1.0,
             "generated_text": original_generated_text,
             "target_text": original_target_text}
        ],
    }
```

Đoạn trên mô tả interface, không phải verifier chạy sẵn. Adapter chịu trách nhiệm giữ nguyên prompt/system/chat-template/decoding của IF; pipeline không thay verifier bằng exact match. Mọi model dùng chung generation defaults từ checkpoint IF FP, cộng cấu hình `generation` truyền vào verifier; defaults được ghi metadata, tránh checkpoint quantized có defaults khác. Cần trả đúng một record cho mỗi query. Không lưu secret key trực tiếp trong YAML/metadata; verifier có thể đọc key từ environment hoặc key file.

## RTN và AWQ

RTN thực thi groupwise theo hàng, nhóm dọc chiều input của matrix, xử lý nhóm cuối ngắn. Symmetric: mã `[-(2^(b-1)-1), +(2^(b-1)-1)]`, scale=maxabs/qmax, không zero-point. Asymmetric: mã `[0, 2^b-1]`, range bao gồm 0, zero-point làm tròn. `numpy.rint` dùng ties-to-even. Đây là RTN fake quantization với trọng số dequantized; không phải kernel inference packed và có thể khác RTN backend ban đầu của bạn. Để tái lập quan sát 1.00/0.75, trước tiên đối chiếu quy ước RTN và Stage 0. Không mặc định coi các con số tham chiếu là kết quả.

AWQ3 dùng **code AWQ vendored trong `vendor/drive_awq` từ Google Drive folder đã cung cấp**, xuất về HF dequantized **trong cùng hệ tọa độ parameter** của model gốc. Repo không gọi RTN rồi gắn nhãn AWQ. Packed `qweight/qzeros` bị từ chối. Phải hoàn nguyên mọi reparameterization/scaling/fusion để `Q(W_F)-Q(W)` có ý nghĩa; chỉ unpack integer codes là chưa đủ. Bảo toàn tokenizer, architecture, module grouping và dùng cùng calibration token data cho clean/IF ở cùng seed.

Mỗi thư mục checkpoint AWQ phải có `metadata.json`:

```json
{
    "quantizer": "awq", "bits": 3, "group_size": 128,
    "symmetric": false, "zero_point": true,
  "calibration_dataset": "public_calibration",
  "calibration_sample_count": 128, "calibration_sequence_length": 2048,
  "calibration_sha256": "SHA256_OF_EXACT_CALIBRATION_TOKEN_DATA",
  "seed": 42, "source_checkpoint": "checkpoints/clean",
  "storage_representation": "hf_dequantized",
  "quantization_library_version": "backend-version-or-commit"
}
```

Các trường phải khớp config, kể cả đường dẫn `source_checkpoint`. Mẫu config dùng `{seed}` trong đường dẫn checkpoint. Nếu calibration được lấy mẫu khác theo seed, tạo output root riêng cho mỗi bộ calibration/config; cấu hình mặc định chạy một seed `[42]` để không lặp quantize/inference vô ích trên Colab.

## Chạy thí nghiệm

Chỉ cần có Python 3.10+ và sửa `configs/common.yaml`, chạy một lệnh để tự cài environment rồi chạy toàn bộ:

```bash
bash run_full.sh
```

Script chỉ dùng Python của environment server hiện tại và không tự cài package, không tạo `.venv`. Cài dependencies một lần bằng `python -m pip install -r requirements.txt`, sau đó chạy FP, RTN3, RTN4 và AWQ3, Stage 0 → Batch A → Batch B → CSV/biểu đồ/report. Có thể chọn Python bằng `PYTHON=/path/to/python bash run_full.sh`. Kết quả tổng hợp nằm trong `outputs/original_if/summary/` theo cấu hình mặc định. Chạy `bash run_full.sh --help` để xem tùy chọn.

Trước khi chạy batch, script tải checkpoint base và IF-SFT, tải IF queries gốc, tạo matched-normal controls từ public Alpaca theo token-length tolerance, tạo calibration từ Wikitext-2 train, sau đó chuẩn bị AWQ3 cho clean/fingerprinted bằng code vendored trong repo. AWQ là artifact cache: nếu checkpoint, manifest và metadata đã đủ thì lần sau chỉ reuse/load ra phân tích, không gọi quantize lại. Sau đó script preflight checkpoint, dữ liệu query, verifier IF và metadata. Có thể chạy riêng validation bằng `phase1 validate --configs configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/awq3.yaml`.

```bash
# Stage 0 + Batch A, đúng thứ tự trên tất cả config:
phase1 batch --configs configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/awq3.yaml

# Sau khi xem tín hiệu Batch A, chạy lại với Batch B:
phase1 batch --configs configs/fp.yaml configs/rtn3.yaml configs/rtn4.yaml configs/awq3.yaml --include-batch-b

# Chỉ phân tích parameter, chưa cần IF verifier:
python scripts/01_parameter_update_retention.py --config configs/rtn3.yaml
python scripts/02_update_vs_quant_step.py --config configs/rtn3.yaml
python scripts/03_error_alignment.py --config configs/rtn3.yaml

# Chạy riêng những stage cần thiết; không cần chạy lại Batch A:
phase1 run --config configs/rtn3.yaml --stages 3,6,7
python scripts/08_compare_quantizers.py --output-root outputs
```

Có đủ scripts `00`–`09` theo spec. Experiment 4 chỉ quét RTN3; bắt đầu từ FP và khôi phục weight trong `finally` sau mỗi block/module. Chọn top 3 layer theo fingerprint drop, không theo ratio gần chia 0. Experiment 5 tự lấy kết quả RTN3 cùng seed; có thể truyền `--rtn3-results 'path/seed{seed}/baseline/if_query_results.csv'`, kèm metadata của run đó. Hidden drift yêu cầu Experiment 4 đã chạy.

## Outputs và cách đọc

Mỗi run mặc định: `outputs/original_if/<fp|rtn3|rtn4|awq3>/seed42/metadata.json` và các thư mục kết quả đúng tên spec. Tách seed để không ghi đè; summary gom vào `outputs/original_if/summary/`, gồm `quantizer_comparison.csv`, `statistics.json`, `phase1_summary.md`, `figures/*.png` và `*.svg`.

- Retention/collision/alignment: tensor, block, module và global. Aggregate L2 bằng căn tổng bình phương; collision dùng tổng count, không trung bình tỷ lệ tensor. Không clip retention >1. Collision không có weight thay đổi để trống.
- Resolution: exact quantiles/fractions cho weight có `|delta| > 1e-8`; scale của clean grid. Crossing dùng **cùng clean grid** cho cả weight clean/IF; collision so giá trị dequantized của hai grid được fit riêng. Khoảng cách boundary chỉ xét boundary nội bộ, đúng cả saturation tails.
- Biểu đồ ECDF/scatter resolution dùng mẫu capped đều theo tensor, được ghi rõ; không phải pooled global distribution. Curve layer resolution là trung bình thống kê tensor.
- Teacher forcing đo target token tại causal-shift đúng. Nhóm survived/failed dựa trên RTN3, còn margin để so nhóm là **FP trước lượng tử hóa**. Xuất thêm token-level CSV.
- PPL là exp(total NLL / scored tokens), chia corpus thành cửa sổ overlap một token; mỗi token trừ token đầu được chấm đúng một lần. Đây là fixed-window PPL, không phải sliding context toàn phần.
- Logit drift đo ở final prompt token: KL(FP||Q), JS, cosine, top-1/5/10. Hidden drift hook trực tiếp output transformer block, lưu final prompt token và mean prompt tokens; không nhầm embedding hoặc final norm là một block.
- Drift chạy cả clean và IF, fingerprint và matched normal prompts. Statistics có paired logit JS difference-in-differences, bootstrap mean/std/median/95% CI; gom repeated seeds theo query trước bootstrap. Survived-vs-failed có Mann–Whitney và Cliff's delta riêng từng seed.
- Metadata ghi config, seed, versions, GPU, thời gian UTC, git commit nếu có, source-code hash và data hashes. Run từ chối trộn output nếu code/config/input thay đổi; chọn output root mới. Checkpoint được ghi bằng path; nếu thay nội dung checkpoint tại cùng path, dùng thư mục/version mới.

`phase1_summary.md` nêu Q1–Q7, dữ liệu còn thiếu và tag access. Báo cáo **không tự tuyên bố** H1–H5 đúng hay Phase 1 nghiên cứu đã thành công: cần checkpoint thực, verifier gốc và đánh giá bằng chứng. Report/plots vẫn chạy khi mới có Batch A hoặc parameter-only, phần thiếu ghi rõ.

Kiến trúc mặc định nhận các block `layers.N`, `h.N`, `blocks.N` và các module q/k/v/o/gate/up/down_proj. Model khác phải cấu hình `module_pattern`, `block_pattern`, `module_aliases` tương ứng. Cấu hình pinned cho kết quả công bố nên khóa môi trường sau cài bằng `pip freeze`; môi trường kiểm thử hiện ghi trong `docs/validation.md`.

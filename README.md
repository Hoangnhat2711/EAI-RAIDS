# RCL-Framework

Đây là một khuôn khổ (framework) cho Học Liên tục được Điều hòa (Regularized Continual Learning) sử dụng PyTorch.

## Cấu trúc dự án

```
.|
├── core/
│   └── rcl_trainer.py      # Logic huấn luyện cốt lõi của framework
├── data/
│   ├── dataset_loaders.py  # Placeholder cho việc tải dữ liệu
│   └── data_transforms.py  # Placeholder cho các phép biến đổi dữ liệu
├── models/
│   ├── base_model.py       # Lớp mô hình cơ sở
│   ├── cnn_model.py        # Mô hình CNN cho dữ liệu phi cấu trúc (ví dụ: hình ảnh)
│   └── mlp_model.py        # Mô hình MLP cho dữ liệu có cấu trúc (ví dụ: dạng bảng)
├── regularization/
│   └── ewc.py              # Triển khai Elastic Weight Consolidation (EWC)
├── main.py                 # File chính để chạy thử nghiệm
├── rcl_framework.py        # File gốc, sẽ được di chuyển và xóa sau
└── requirements.txt        # Các thư viện Python cần thiết
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

Để chạy ví dụ minh họa của framework, hãy thực thi:

```bash
python main.py
```

## Ghi chú

Hiện tại, framework sử dụng dữ liệu giả định để minh họa luồng hoạt động. Các file trong `data/` chứa các placeholder cho việc tích hợp dữ liệu thực tế.

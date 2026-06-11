# Tài liệu Hướng dẫn & Giải thích Dự án: Olist Early Warning System

## 1. Giới thiệu Bài toán (Problem Statement)
**Bối cảnh:** Olist là một nền tảng thương mại điện tử lớn tại Brazil. Đóng vai trò là trung gian kết nối người bán và người mua. Khi khách hàng có trải nghiệm tồi tệ (giao trễ, phí ship đắt, v.v.), họ thường để lại đánh giá thấp (1, 2, 3 sao). Việc phát hiện muộn khiến công ty mất đi cơ hội chăm sóc khách hàng kịp thời.

**Mục tiêu:** Xây dựng một mô hình Machine Learning **Phân loại nhị phân (Binary Classification)** đóng vai trò như một **Hệ thống Cảnh báo sớm (Early Warning System)**. 
- **Thời điểm kích hoạt:** Ngay khi hệ thống ghi nhận đơn hàng vừa chuyển sang trạng thái `delivered` (đã giao hàng thành công).
- **Dự đoán:** Khách hàng này có nguy cơ để lại đánh giá xấu hay không, dựa trên toàn bộ dữ liệu lịch sử của đơn hàng đó (thời gian vận chuyển, thông tin sản phẩm, cách thanh toán, v.v.).
- **Biến mục tiêu (Target):** 
  - `Class 1 (Tốt)`: Đánh giá 4, 5 sao.
  - `Class 0 (Xấu - Cần cảnh báo)`: Đánh giá 1, 2, 3 sao.

---

## 2. Những gì chúng ta đã làm (Methodology & Implementation)
Để giải quyết bài toán trên một cách tối ưu và không bị sai lệch logic dữ liệu (Data Leakage), project đã được triển khai qua các bước chuẩn chỉnh sau:

### 2.1. Gom cụm và Làm sạch Dữ liệu (Aggregation & Cleaning)
Dữ liệu gốc phân tán ở 9 bảng khác nhau với nhiều "cạm bẫy" (quan hệ 1-nhiều). Chúng ta đã:
- **Xử lý quan hệ 1-nhiều:** `Groupby` bảng `Items` (chi tiết hàng hóa) và `Payments` (thanh toán) theo mã đơn hàng (`order_id`) để tính tổng số tiền (`SUM`), tổng phí ship, và số lượng trả góp trước khi ghép bảng.
- **Xử lý trùng lặp:** Loại bỏ các tọa độ lặp lại của cùng một mã Zipcode, và chỉ giữ lại đánh giá (Review) mới nhất nếu khách hàng cập nhật đánh giá nhiều lần.

### 2.2. Feature Engineering (Tạo đặc trưng quan trọng)
Máy học cần những con số mang ý nghĩa nghiệp vụ. Chúng ta đã tự tính toán thêm:
- **`delivery_time` (Thời gian giao hàng):** Khoảng thời gian từ lúc khách bấm đặt hàng đến khi nhận được hàng.
- **`delay_time` (Thời gian trễ):** Khoảng thời gian khách phải chờ thêm so với ngày giao hàng dự kiến ban đầu (nếu giao sớm thì bằng 0). Đây là yếu tố cực kỳ quan trọng ảnh hưởng đến cảm xúc khách hàng.

### 2.3. Hợp nhất Dữ liệu (Building Master Table)
Sử dụng bảng `Orders` (chỉ lấy đơn hàng `delivered`) làm trái tim, sau đó `LEFT JOIN` tuần tự với tất cả các bảng dữ liệu đã làm sạch ở trên để tạo thành một bảng tổng hợp (Master Table). Mỗi dòng đại diện cho **duy nhất 1 đơn hàng**.

### 2.4. Tiền xử lý & Cân bằng Dữ liệu (Preprocessing & SMOTE)
- **Chuẩn hóa (Scaling & Encoding):** Đưa các biến dạng phân loại (như Tên tiểu bang) về dạng ma trận 0-1 (One-Hot Encoding). Chuẩn hóa các biến số liên tục (như giá tiền, thời gian) về cùng một thang đo bằng `StandardScaler`.
- **Giải quyết Mất cân bằng lớp (Class Imbalance):** Số lượng khách hàng đánh giá xấu (Class 0) chiếm tỷ lệ rất nhỏ so với khách hài lòng. Nếu không xử lý, AI sẽ có xu hướng đoán toàn bộ là "Tốt". Chúng ta đã sử dụng thuật toán **SMOTE** để sinh ra các mẫu dữ liệu ảo cho Class 0 trên tập Huấn luyện (Train set), giúp mô hình học cách nhận biết dấu hiệu chê bai tốt hơn.

### 2.5. Huấn luyện và Đánh giá Mô hình (Modeling & Evaluation)
Sử dụng 4 thuật toán phổ biến từ cơ bản đến nâng cao:
1. **Logistic Regression** (Mô hình tuyến tính cơ bản)
2. **Random Forest** (Mô hình tập hợp cây quyết định)
3. **XGBoost** (Mô hình Boosting mạnh mẽ)
4. **LightGBM** (Mô hình Boosting tốc độ cao, cực kỳ phù hợp cho dữ liệu lớn)

**Tiêu chí đánh giá:** Vì đây là bài toán "Cảnh báo sớm", chúng ta không chỉ nhìn vào độ chính xác tổng thể (Accuracy) mà tập trung vào **Recall của Class 0** (Khả năng bắt trúng những người có nguy cơ chê bai mà không bị bỏ sót) và chỉ số **ROC-AUC**. Cuối cùng, trực quan hóa mức độ quan trọng của các biến số (Feature Importance) để rút ra Insights cho doanh nghiệp.

---

## 3. Cấu trúc Dự án
- `input/`: Chứa bộ dữ liệu gốc của Olist.
- `Olist_Early_Warning_System.ipynb`: File Jupyter Notebook chứa toàn bộ mã nguồn Python thực thi các bước nêu trên. File được phân chia rõ ràng từng khối lệnh, kèm văn bản giải thích.
- `requirements.txt`: Danh sách các thư viện Machine Learning cần thiết.
- `README.md`: File hướng dẫn này.
- Các file tài liệu `.md` khác: Phục vụ cho việc tóm tắt và phân tích từ điển dữ liệu.

---

## 4. Hướng dẫn Chạy (How to run)
**Bước 1: Cài đặt thư viện**
Mở Terminal/Command Prompt, trỏ tới thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```
*(Nếu bạn đã cài đặt qua hướng dẫn trước đó thì có thể bỏ qua bước này).*

**Bước 2: Mở và chạy File Code**
- Mở file `Olist_Early_Warning_System.ipynb` bằng VS Code hoặc Jupyter Notebook.
- Đảm bảo chọn đúng Kernel Python (môi trường mà bạn vừa cài đặt thư viện).
- Bấm **Run All** (Chạy tất cả) và cuộn từ trên xuống dưới để xem trực tiếp các biểu đồ phân tích và kết quả dự đoán của các mô hình AI.

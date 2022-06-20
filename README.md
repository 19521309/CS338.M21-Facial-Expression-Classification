# CS338.M21 - Facial Expression Recognition

## Nhóm 10

**Giảng viên viên hướng dẫn :  Đỗ Văn Tiến**
| MSSV       |  Họ và Tên       | Công Việc                               |
| -----------| -------------    |-----------------------------------------|
| 19522454   | Nguyễn Tấn Tú    | [Train ResNet50](https://www.kaggle.com/code/tunguyentan/face-emotion-recognition-using-resnet50), [Detect Face and Predict](https://colab.research.google.com/drive/1Usju5dw62w1DWohTcj_Q7UezgPcG2j1S?usp=sharing) |
| 19522555   | Nguyễn Thị Như Ý | [Train ResNet101](https://www.kaggle.com/code/ynguyenntc/face-emotion-recognition-using-resnet101/notebook?scriptVersionId=98631927), làm slide PowerPoint   |
| 19521217   | Trần Nguyễn Quỳnh Anh | [Train ResNet152](https://www.kaggle.com/code/anhtrnnguynqunh/face-emotion-recognition-using-resnet152), làm slide PowerPoint|
| 19521309   | Đinh Hoàng Linh Đan | [Deploy web API](https://drive.google.com/file/d/1Q_dHGH5G9k5SHwUfKCHIDOgZ0fHA3dNI/view?usp=sharing) |
#### Kết quả nhận được của 3 model sau khi train, [here](https://drive.google.com/drive/folders/1i6WXwbLw936VR_8g6qYCzWL8z_WaHlAN?usp=sharing)

## Detect Face and Predict
Detect khuôn mặt sử dụng thư viện của OpenCV. Với các tham số scaleFactor  = 1.05, minNeighbors =  6, minSize = (int(x_min), int(y_min))
Với, original là ảnh input.
*x_min = original.shape[0]/15 , 1/15 chiều cao của ảnh.
y_min = original.shape[1]/15 , 1/15 chiều rộng của ảnh* 
     
***Source code*** của Detect Face and Predict trong file DetectFaceAndPredict.ipynb

## Ứng dụng dự đoán biểu cảm gương mặt sử dụng Streamlit, Python dựa trên 3 model đã được train ở trên.
***Source code*** của Deploy Web API trong file streamlit.py.

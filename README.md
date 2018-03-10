# OCR
自动识别[武汉科技大学教务处](http://jwxt.wust.edu.cn/whkjdx/)登录时的[验证码](http://jwxt.wust.edu.cn/whkjdx/verifycode.servlet?0.12337475696465894)  
单个字符识别准确率达99.2%


文件说明:  
data文件为训练文件存放的地方，其下的图片按照标签分为不同的文件夹  
verifycode为测试时验证码保存的文件夹  
x_train.pickle为X_train的序列化文件  
y_train.pickle为y_train的序列化文件  
wust.ocr为class OCR()的序列化文件  


## 识别流程
大致思路是去除图片的干扰→ 将图片二值化→ 将图片分割为单个字符→ 提取单个字符的特征→ 生成训练集→ 构建模型进行训练→ 训练完成进行预测

### 处理获取到的图片
通过del_blur()函数去除图片的噪声并将图片二值化
```python
def del_blur(img):
	# 双边滤波 去噪 效果不错
	m_blur = cv2.bilateralFilter(img, 9, 75, 75)
	# oust 滤波 二值化图片
	ret, oust_img = cv2.threshold(m_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	fin = cv2.bilateralFilter(oust_img, 9, 75, 75)
	return fin

```
### 图片分割
通过切片方式截取到单个字符，截取到的字符需要长宽相等
具体位置是通过多次尝试得出
```python
def split_img(img):
	hs = 3, h = 14, w = 11
	ws1 = 3, ws2 = 13
	ws3 = 23, ws4 = 33
	img1 = img[hs:hs + h, ws1:ws1 + w]
	img2 = img[hs:hs + h, ws2:ws2 + w]
	img3 = img[hs:hs + h, ws3:ws3 + w]
	img4 = img[hs:hs + h, ws4:ws4 + w]
	return img1, img2, img3, img4

```

### 获取图片特征
[HOG特征](https://www.jianshu.com/p/d3f93c360226)  
[图片的其他特征](http://dataunion.org/20584.html)

```python
from skimage import feature as ft

def hog(img):
	"""
	提取图片hog特征（将图片信息降维 二维数组变一维数组）
	:param img: 图片文件的地址或numpy.array对象
	:return f: 图片hog特征数据（一维）
	"""
	if isinstance(img, str):
		img = Image.open(img)
	elif isinstance(img, np.ndarray):
		img = Image.fromarray(img)
	else:
		raise ValueError('the input img is not fill the bill! \n '
						 'it must be an image file path or numpy.ndarray')

	return ft.hog(img, block_norm='L2-Hys', pixels_per_cell=(2, 2), cells_per_block=(2, 2))

```
可以通过改变ft.hog的pixels_per_cell、cells_per_block两个参数来改变特征信息的长度，
由于分割得到的图片尺寸(14×11)较小我这里的参数也设置的较小,当图片较大时可适当扩大参数，避免特征信息长度过长而造成训练数据过大

### 生成训练数据

```python

X_train=[]
y_train=[]
for label in os.listdir(r'data'):
	label_path = os.path.join(os.path.abspath('.'), 'data', label)
	label_items = os.listdir(label_path)
	for item_name in label_items:
		item = os.path.join(label_path, item_name)
		feature = hog(item)
		X_train.append(feature)
		y_train.append(label)

```
X_train为图片的特征信息，y_train为图片的标签与X_train通过下标相对应

### 构建并训练模型
构建SVM模型对数据进行分类
```python
#构建svm模型
svm = SVC(kernel='rbf', random_state=0, gamma='auto', C=1.0, probability=True)
#将数据划分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
#将数据进行标准化处理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#训练模型
svm.fit(X_train_std, y_train)
#预测，并获取分数
y_pred = svm.predict(X_test_std)
score = accuracy_score(y_test, y_pred)
```

## 参考数据
这本书讲的通俗易懂很适合入门  
[Python机器学习-豆瓣](https://book.douban.com/subject/27000110/)  
[大佬的学习笔记](https://ljalphabeta.gitbooks.io/python-/content/)  
![](https://img1.doubanio.com/lpic/s29407827.jpg)

## requirements
* opencv
* numpy
* scikit-image
* scikit-learn

通过whl安装OpenCV和scikit-image（通过pip安装这个两个模块可能会报错）
> [OpenCV](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)  
> [scikit-image](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)

通过pip 安装scikit-learn和numpy
> pip install scikit-learn
> pip install numpy













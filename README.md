# Python-Machine-Learning-Mini-Course-zh-TW

[Python Machine Learning Mini-Course](https://machinelearningmastery.com/python-machine-learning-mini-course/) 是 Jason Brownlee 在 Machine Learning Mastery 所發表的一篇教學文，內容淺顯易懂，對於想要透過 Python 來入手機器學習的新手來說，是很好的文章，在這裡分享給大家。

## 14 天從開發者到機器學習工作者

Python 已經成為應用機器學習領域中發展最快速的平台語言之一。

在這門簡短的課程中，你會了解到如何在 14 天內，使用 Python 來建構機器學習模型，並且有自信地完成一個機器學習的專案。

這是一篇很重要的文章，你可以把他加入書籤中。

讓我們開始吧！

[更新] Oct/2016: 更新範例至 sklearn v0.18
[更新] Feb/2018: 更新 Python 和 library 版本
[更新] March/2018: 更改某些資料集的下載連結，某些連結已經失效

## 這門課程的對象是誰？

在我們開始學習之前，讓我們確保你站在正確的位置。

底下我描述了一些通則，讓你知道這門課所設計的學習對象為何。

如果你不完全符合以下的條件，別緊張，你可能只要在某一個或幾個領域重新學習就可以跟上。

- 知道如何撰寫一些程式碼的開發者。這代表說你在學習一門新的語言，像是 Python 時，一旦你知道了基本的語法，這不是太大的問題。這不代表你需要是一個開發狂熱份子，只要你可以輕鬆的了解基本的 C 或類似於 C 語言的語法即可。
- 知道一點點機器學習相關知識的開發者。意味者你知道基本的 cross-validation、一些演算法和偏差和方差之間的取捨 (Bias–variance tradeoff) 等概念。

這門基本的課程並不是 Python 或機器學習的教科書。

這門課程會讓你從一個知道一點點機器學習的開發者，成長為一個能使用 Python 相關生態系來得到機器學習所訓練出來的模型結果。

## 課程導覽

Python Machine Learning Mini-Course
by Jason Brownlee on September 26, 2016 in Python Machine Learning
From Developer to Machine Learning Practitioner in 14 Days
Python is one of the fastest-growing platforms for applied machine learning.

In this mini-course, you will discover how you can get started, build accurate models and confidently complete predictive modeling machine learning projects using Python in 14 days.

This is a big and important post. You might want to bookmark it.

Let’s get started.

Update Oct/2016: Updated examples for sklearn v0.18.
Update Feb/2018: Update Python and library versions.
Update March/2018: Added alternate link to download some datasets as the originals appear to have been taken down.
Python Machine Learning Mini-Course
Python Machine Learning Mini-Course
Photo by Dave Young, some rights reserved.

Who Is This Mini-Course For?
Before we get started, let’s make sure you are in the right place.

The list below provides some general guidelines as to who this course was designed for.

Don’t panic if you don’t match these points exactly, you might just need to brush up in one area or another to keep up.

Developers that know how to write a little code. This means that it is not a big deal for you to pick up a new programming language like Python once you know the basic syntax. It does not mean you’re a wizard coder, just that you can follow a basic C-like language with little effort.
Developers that know a little machine learning. This means you know the basics of machine learning like cross-validation, some algorithms and the bias-variance trade-off. It does not mean that you are a machine learning Ph.D., just that you know the landmarks or know where to look them up.
This mini-course is neither a textbook on Python or a textbook on machine learning.

It will take you from a developer that knows a little machine learning to a developer who can get results using the Python ecosystem, the rising platform for professional machine learning.

Need help with Machine Learning in Python?
Take my free 2-week email course and discover data prep, algorithms and more (with code).

Click to sign-up now and also get a free PDF Ebook version of the course.

Start Your FREE Mini-Course Now!
Mini-Course Overview
This mini-course is broken down into 14 lessons.

You could complete one lesson per day (recommended) or complete all of the lessons in one day (hard core!). It really depends on the time you have available and your level of enthusiasm.

Below are 14 lessons that will get you started and productive with machine learning in Python:

- Lesson 1: Download and Install Python and SciPy ecosystem.
- Lesson 2: Get Around In Python, NumPy, Matplotlib and Pandas.
- Lesson 3: Load Data From CSV.
- Lesson 4: Understand Data with Descriptive Statistics.
- Lesson 5: Understand Data with Visualization.
- Lesson 6: Prepare For Modeling by Pre-Processing Data.
- Lesson 7: Algorithm Evaluation With Resampling Methods.
- Lesson 8: Algorithm Evaluation Metrics.
- Lesson 9: Spot-Check Algorithms.
- Lesson 10: Model Comparison and Selection.
- Lesson 11: Improve Accuracy with Algorithm Tuning.
- Lesson 12: Improve Accuracy with Ensemble Predictions.
- Lesson 13: Finalize And Save Your Model.
- Lesson 14: Hello World End-to-End Project.

Each lesson could take you 60 seconds or up to 30 minutes. Take your time and complete the lessons at your own pace. Ask questions and even post results in the comments below.

The lessons expect you to go off and find out how to do things. I will give you hints, but part of the point of each lesson is to force you to learn where to go to look for help on and about the Python platform (hint, I have all of the answers directly on this blog, use the search feature).

I do provide more help in the early lessons because I want you to build up some confidence and inertia.

## 第一課：下載並安裝 Python 和 SciPy

你必須要擁有 Python 相關開發環境後，才能夠開始學習機器學習。

今天的課程很簡單，你必須要下載 Python 3.6 在自己的電腦上。

訪問 Python 官方網站，並根據自己的作業系統下載對應的 Python 版本。你可能要根據自己的作業系統來使用對應的套件管理工具來進行安裝，像是 OSX 的 macports 或 RedHat 的 yum。

你也需要安裝 SciPy 以及 scikit-learn，我建議使用和安裝 Python 相同的方式來進行安裝。

你也可以使用 Anaconda，它幫你把所有需要的函式庫都打包在一起，對於初學者來說相當方便。

開始學習 Python 了，你可以在 command line 輸入 `python` 指令來進入 python 的互動 shell。

透過以下簡單的程式碼來確認你的函式庫的版本：

```python
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
```

如果出現任何錯誤，現在就花時間修正他們。

需要幫助的時候，看看底下的這篇文章：

[如何使用 Anaconda 來建立機器學習和深度學習的開發環境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 第二課：開始練習 Python、NumPy、Matplotlib 和 Pandas

你需要有能力可以撰寫基本的 Python 程式碼。

身為開發者，你可以快速的學習新的程式語言。Python 區分大小寫，# 作為註解，並使用空白來表示程式碼區塊 (空白是重要的)。

今天的任務是在 Python 的互動環境中練習 Python 的基本語法，並練習 SciPy 的資料結構。

練習使用指派運算，使用 list 和流程控制。
練習 NumPy 陣列。
練習使用 Matplotlib 建立簡單的圖表。
練習使用 Pandas 的 Series 和 DataFrames。
舉例來說，底下是使用 Pandas 建立 DataFrame 的例子：

```python
# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
```

## 第三課：從 CSV 中讀取資料

機器學習演算法需要資料。你的資料可以從自己的資料集來，可是當你一開始練習的時候，應該練習使用標準機器學習資料集。

今天你的任務就是熟悉在 Python 中讀取滋藥，並且使用標準的機器學習資料集。

在網路你可以找到許多優秀的機器學習用的資料集。在這堂課中，你可以從[UCI 機器學習資料庫](http://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/) 中下載相關的資料集。

練習在 Python 中使用 [CSV.reader()](https://docs.python.org/2/library/csv.html) 來讀取 CSV 資料。

練習使用 Numpy 的 [numpy.loadtxt()](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html) 函示來讀取資料。

練習使用 Pandas 的 [pandas.read_csv()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) 函示來讀取資料。

為了讓你可以更快入門，下面的程式碼是一個簡單的範例，它直接從 UCI 機器學習資料庫中，透過 Pandas 來讀取 Pima 印地安人糖尿病資料集：

```python
    # Load CSV using Pandas from URL
    import pandas
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = pandas.read_csv(url, names=names)
    print(data.shape)
```

## 第四課：透過敘述統計的方法來瞭解資料

一旦你將資料讀取到 Python 後，你就能夠更好地瞭解你的資料。

當你更了解你的資料，你所建立的模型就會更好且更準確。了解資料的第一步就是使用敘述統計的方式來進行。

今天的功課就是透過敘述統計的方法來了解資料。我建議你可以使用 Pandas DataFrame 提供的相關函式。

透過 head() 函式來看看數筆資料。透過 shape 屬性來觀看資料的維度。透過 dtypes 屬性來看看資料的型態。透過 describe() 函式來看看資料的分佈狀況。使用 corr() 函式來計算資料之間的相關係數。

底下的範例是讀取 Pima 印地安人糖尿病資料集後，透過 describe() 函式來觀看資料的分佈。

```python
    import pandas
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = pandas.read_csv(url, names=names)
    description = data.describe()
    print(description)
```

## 第五課：透過視覺化來了解資料

延續昨天的課程，你必須要花時間了解你的資料。

第二種了解資料的方式就是透過視覺化的技巧(例如：繪圖)。

今天，你的功課就是去學習如何在 Python 中透過繪圖的方法來了解資料中個屬性的特性，以及彼此交互的關係。同樣，我也建立可以使用 Pandas 中相關輔助的函式來幫助你更了解資料。

使用 hist() 函式來建立每個屬性的直方圖。
使用 plot(kind='box') 函式來建立每個屬性的 box-and-whisker 圖。
使用 pandas.scatter_matrix() 函式來建立所有屬性兩兩之間的散佈圖。

舉例來說，底下的程式碼會讀取糖尿病資料集，並建立資料集之間的散佈圖矩陣。

```python
    # Scatter Plot Matrix
    import matplotlib.pyplot as plt
    import pandas
    from pandas.plotting import scatter_matrix
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = pandas.read_csv(url, names=names)
    scatter_matrix(data)
    plt.show()
```

![image](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/09/Sample-Scatter-Plot-Matrix.png)


## 第六課：針對資料進行前處理，準備進入建模階段

你的原始資料很有可能不是建立模型的最好狀態。

很多時候，你需要針對資料進行前處理，讓你的資料可以很好的餵給模型演算法。今天的課程中，你將會使用 scikit-learn 的資料前處理功能來處理資料。

scikit-learn 的函式庫提供了兩種標準的資料轉換方式。這兩種都很有用，分別是：進行 fit() 和 多次的 transform()，或是 fit() 和 transform() 結合的轉換。

有許多的方法可以在建模前來進資料準備，讓我們看看底下的例子：

- 對數值資料進行標準化 (例如：將資料轉換為平均數為 0，標準差為 1 的分配)
- 利用 range 的參數讓數值資料標準化 (例如：將數值資料轉換為 0-1 的區間)
- 探索更進階的資料工程技巧，例如：二值化 (Binarizing)

來看個例子，底下的程式碼會讀取 Pima 印地安人糖尿病資料集，計算進行資料正規化所需要的參數，然後建立正規化後的資料：

```python
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```

## 第七課：透過重複抽樣(Resample method)的方法來進行演算法評估

用來給機器學習演算法學習的資料集稱之為訓練資料集。然而，訓練資料集並不能保證機器學習演算法學習到的模型能夠完美的用來預測新的資料。這就是一個大的問題，因為我們之所以訓練模型，就是希望能夠準確的預測新的資料。

要解決這樣的問題，你可以透過一種統計的方法，叫做重複抽樣，將你的訓練資料集分成數個子集合，某些用來訓練模型，其他的則是用來評估模型的準確度，以便了解訓練出來的模型在面對沒有看過的資料時的效果如何。

而今天課程的目的就是要來練習這種重複抽樣的方法，在 scikit-learn 當中，你可以透過以下步驟來實現：

- 將資料集分成訓練資料集和測試資料集
- 透過 k-fold 交叉驗證的方法來預估某個模型的準確率
- 透過 leave one out 交叉驗證的方式來預估某個演算法的準確率

The snippet below uses scikit-learn to estimate the accuracy of the Logistic Regression algorithm on the Pima Indians onset of diabetes dataset using 10-fold cross validation.

底下的程式碼使用 scikit-learn 的 10-fold 交叉驗證的方式來驗證使用 Logistic Regression 演算法在 Pima 印地安人糖尿病資料集上的準確率。

```python
# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
```
What accuracy did you get? Let me know in the comments.

你得到多少的準確率？在底下留言讓我知道。

你知道目前已經學習到一半了嗎？做得好！

## Reference
- [Python Machine Learning Mini-Course](https://machinelearningmastery.com/python-machine-learning-mini-course/)

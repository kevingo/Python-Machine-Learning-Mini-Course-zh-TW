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

## Lesson 5: Understand Data with Visualization

Continuing on from yesterday’s lesson, you must spend time to better understand your data.

A second way to improve your understanding of your data is by using data visualization techniques (e.g. plotting).

Today, your lesson is to learn how to use plotting in Python to understand attributes alone and their interactions. Again, I recommend using the helper functions provided on the Pandas DataFrame.

Use the hist() function to create a histogram of each attribute.
Use the plot(kind=’box’) function to create box-and-whisker plots of each attribute.
Use the pandas.scatter_matrix() function to create pairwise scatterplots of all attributes.
For example, the snippet below will load the diabetes dataset and create a scatterplot matrix of the dataset.

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

## Reference
- [Python Machine Learning Mini-Course](https://machinelearningmastery.com/python-machine-learning-mini-course/)

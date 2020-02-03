import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from PIL import Image

data = pd.read_csv('./Market_Basket_Optimisation.csv')
# 数据预处理
data.fillna(0, inplace=True)


def get_word_cloud(df, top=10):
    sentences = list(df.values)
    words = []
    for sentence in sentences:
        for word in sentence:
            if word != 0:
                words.append(word)
    text = ' '.join(words)
    heart_mask = np.array(Image.open('heart_mask.png'))
    wc = WordCloud(background_color="white", max_words=top, mask=heart_mask,
                   max_font_size=100, contour_width=2,  # 设置轮廓宽度
                   contour_color='steelblue')  # 设置轮廓颜色
    wordcloud = wc.generate(text)
    return wordcloud


jpg = get_word_cloud(data, top=10)
plt.imshow(jpg)
plt.axis("off")
plt.show()
jpg.to_file('test4.png')

import mlflow
import pandas as pd
from vk_api import VkApi

# Функция для парсинга постов со стены выбранной группы ВК через group_id
def main(offset: int, token: str, group_id: str):
    vk = VkApi(token=token) # авторизация через токен 
    api = vk.get_api()
    posts = api.wall.get(owner_id = group_id, offset = offset, count=100)['items']
    posts_strings = [post['text'] for post in posts]
    return posts_strings

# access_token, полученный из адресной строки
token='vk1.a.xo7XZNad2i4OVf6RwUrDjlBNgeMdwGOHZcR-1ceTw0fNh_EsqlC7ayriui4K79jiK5pQsjL-o-H-4ZT1dV7V7VJHPd6Ep1wcr-I9sWY0WKTKr0BonLEt2nakGuJjUSdra_rr8GsvFEyfL_QvPyzy2bpSveaq1UrCd-Cyl9Vko56nUb4g34LO7eqivzGVHTMpEVByyHQguqq0J4fjFsIWMQ'
# Парсить мне нужно только тексты постов, буду использовать группу **Российские железные дороги (ОАО "РЖД")** в качестве источника.
# РЖД 
group_id='-38981315'

combo_list_posts = []

for i in range(0, 9000, 100):
    try:
        rzd_posts = main(offset = i, token = token, group_id = '-38981315')
        combo_list_posts.extend(rzd_posts)
    except:
        print('Постов больше нет на смещении: ', i)
        

df = pd.DataFrame(data = combo_list_posts, columns=["Text"])

df.to_csv('data/Posts.csv', sep=';', encoding='utf-8', index=False)


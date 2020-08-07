import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


books_1 = pd.read_csv('CSV/book1-100k.csv', engine = 'python', encoding = 'latin-1')
books_2 = pd.read_csv('CSV/book100k-200k.csv', engine = 'python', encoding = 'latin-1')
books_3 = pd.read_csv('CSV/book200k-300k.csv', engine = 'python', encoding = 'latin-1')
books_4 = pd.read_csv('CSV/book300k-400k.csv', engine = 'python', encoding = 'latin-1')
books_5 = pd.read_csv('CSV/book400k-500k.csv', engine = 'python', encoding = 'latin-1')
books_6 = pd.read_csv('CSV/book500k-600k.csv', engine = 'python', encoding = 'latin-1')
books_7 = pd.read_csv('CSV/book600k-700k.csv', engine = 'python', encoding = 'latin-1')
books_8 = pd.read_csv('CSV/book700k-800k.csv', engine = 'python', encoding = 'latin-1')
books_9 = pd.read_csv('CSV/book800k-900k.csv', engine = 'python', encoding = 'latin-1')
books_10 = pd.read_csv('CSV/book900k-1000k.csv', engine = 'python', encoding = 'latin-1')
books_11 = pd.read_csv('CSV/book1000k-1100k.csv', engine = 'python', encoding = 'latin-1')
books_12 = pd.read_csv('CSV/book1100k-1200k.csv', engine = 'python', encoding = 'latin-1')
books_13 = pd.read_csv('CSV/book1200k-1300k.csv', engine = 'python', encoding = 'latin-1')
books_14 = pd.read_csv('CSV/book1300k-1400k.csv', engine = 'python', encoding = 'latin-1')
books_15 = pd.read_csv('CSV/book1400k-1500k.csv', engine = 'python', encoding = 'latin-1')
books_16 = pd.read_csv('CSV/book1500k-1600k.csv', engine = 'python', encoding = 'latin-1')
books_17 = pd.read_csv('CSV/book1600k-1700k.csv', engine = 'python', encoding = 'latin-1')
books_18 = pd.read_csv('CSV/book1700k-1800k.csv', engine = 'python', encoding = 'latin-1')

books = pd.concat([books_1,books_2,books_3,books_4,books_5,books_6,books_7,books_8,books_9,books_10,books_11,books_12,books_13,books_14,books_15,books_16,books_17,books_18]).reset_index(drop = True)

ratings_1 = pd.read_csv('CSV/user_rating_0_to_1000.csv', engine = 'python', encoding = 'latin-1')
ratings_2 = pd.read_csv('CSV/user_rating_1000_to_2000.csv', engine = 'python', encoding = 'latin-1')
ratings_3 = pd.read_csv('CSV/user_rating_2000_to_3000.csv', engine = 'python', encoding = 'latin-1')
ratings_4 = pd.read_csv('CSV/user_rating_3000_to_4000.csv', engine = 'python', encoding = 'latin-1')
ratings_5 = pd.read_csv('CSV/user_rating_4000_to_5000.csv', engine = 'python', encoding = 'latin-1')
ratings_6 = pd.read_csv('CSV/user_rating_5000_to_6000.csv', engine = 'python', encoding = 'latin-1')

Rat_train = pd.concat([ratings_1,ratings_2,ratings_3,ratings_4,ratings_6])
Rat_test = ratings_5

bb = books.drop_duplicates('Name', keep = 'first')
bb['book_id'] = bb.index

Rat_train_1 = pd.merge(Rat_train, bb[['Name', 'book_id']], on='Name')
Rat_test_1 = pd.merge(Rat_test, bb[['Name', 'book_id']], on='Name')

Rat_train_2=Rat_train_1.sort_values('ID').reset_index(drop = True)
Rat_test_2=Rat_test_1.sort_values('ID').reset_index(drop = True)

Rat_train_2.drop('Name', axis =1)
Rat_test_2.drop('Name', axis =1)

Rat_train_2 = Rat_train_2.rename(columns={'ID': 'user_id'})
Rat_test_2 = Rat_test_2.rename(columns={'ID': 'user_id'})

Rat_train_2 = Rat_train_2.reindex(columns= ['user_id','book_id','Rating' ])
Rat_test_2 = Rat_test_2.reindex(columns= ['user_id','book_id','Rating' ])

Rat_train_3 = Rat_train_2.replace({'really liked it': 4, 'it was amazing': 5, 'liked it': 3,'it was ok': 2,'did not like it':1})
Rat_test_3 = Rat_test_2.replace({'really liked it': 4, 'it was amazing': 5, 'liked it': 3,'it was ok': 2,'did not like it':1})

nb_books = max(bb.index)

def convert(data):
  min_users = min(data['user_id'])
  nb_users = max(data['user_id'])
  new_data = np.array([])
  for id_users in range(min_users, nb_users +1):
    id_books = data['book_id'].loc[data['user_id'] == id_users].to_numpy()
    id_ratings = data['Rating'].loc[data['user_id'] == id_users].to_numpy()
    ratings = np.zeros(nb_books)
    ratings[id_books - 1] = id_ratings
    new_data1 = np.append(new_data, ratings )
    new_data = new_data1.reshape(id_users-min_users+1,nb_books )
  return new_data

final_test = convert(Rat_test_3)
#final_training = convert(Rat_train_3)

final_test.to_csv('CSV/R_test.csv')
#final_training.to_csv('CSV/R_train.csv')

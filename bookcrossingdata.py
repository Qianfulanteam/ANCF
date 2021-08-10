import scipy.sparse as sp
import numpy as np


def load_rating_file_as_list():
    ratingList = []
    with open('book_crossing/user_book.test.rating', "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split()
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file():
    negativeList = []
    with open('book_crossing/user_book.test.negative', "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split()
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_train_as_matrix():
        # Get number of users and items
    num_users, num_items = 0, 0
    with open("book_crossing/user_book.info", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            if(arr[1].replace("\n", "") == 'users'):
                num_users = int(arr[0])
            if (arr[1].replace("\n", "") == 'items'):
                num_items = int(arr[0])
            line = f.readline()

    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open('book_crossing/user_book.train.rating', "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split()
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    return mat

def load_itemGenres_as_matrix():
    itemsAttributes = []
    num_items = 0
    dictGender = {}
    genderTypes = 0
    dictAge = {}
    ageTypes = 0
    dictOccupation = {}
    ocTypes = 0

    with open('book_crossing/book_shuxing.txt', "r") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split('::')
            l = []
            for x in arr[0:4]:
                l.append(x)
            itemsAttributes.append(l)
            line = f.readline().strip('\n')
    itemsAttrMat = np.array(itemsAttributes)

    num_items = len(itemsAttrMat)

    # one-hot encoder

    # age types

    genders = set(itemsAttrMat[:, 1])
    for gender in genders:
        dictGender[gender] = genderTypes
        genderTypes += 1

    # age types
    ages = set(itemsAttrMat[:, 2])
    for age in ages:
        dictAge[age] = ageTypes
        ageTypes += 1

    # occupation types
    ocs = set(itemsAttrMat[:, 3])
    for oc in ocs:
        dictOccupation[oc] = ocTypes
        ocTypes += 1

    # Gender,Age,Occupation
    gendermat = sp.dok_matrix((num_items + 1, genderTypes), dtype=np.float32)
    agemat = sp.dok_matrix((num_items + 1, ageTypes), dtype=np.float32)
    occupationmat = sp.dok_matrix((num_items + 1, ocTypes), dtype=np.float32)

    with open("book_crossing/book_shuxing.txt") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split("::")
            userid = int(arr[0])
            usergender = arr[1]
            userage = arr[2]
            useroc = arr[3]
            # gender encoder
            if usergender in dictGender.keys():
                gendermat[userid, dictGender[usergender]] = 1.0
            # age encoder
            if userage in dictAge.keys():
                agemat[userid, dictAge[userage]] = 1.0
            # occupation encoder
            if useroc in dictOccupation.keys():
                occupationmat[userid, dictOccupation[useroc]] = 1.0

            line = f.readline().strip('\n')
        user_gender_mat = gendermat.toarray()
        user_age_mat = agemat.toarray()
        user_oc_mat = occupationmat.toarray()

    # concatenate Gender[0-1], Age[], Occupation
    onehotUsers = np.hstack((user_gender_mat, user_age_mat, user_oc_mat))
    return num_items, onehotUsers



def load_user_attributes():
    usersAttributes = []
    num_users = 0
    #dictGender = {}
    #genderTypes = 0
    # dictAge = {}
    # ageTypes = 0
    # dictOccupation = {}
    # ocTypes = 0
    dictlocation = {}
    locationTypes = 0
    dictAge = {}
    ageTypes = 0

    with open('book_crossing/user_shuxing.txt', "r") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split('::')
            l = []
            for x in arr[0:3]:
                l.append(x)
            usersAttributes.append(l)
            line = f.readline().strip('\n')
    usersAttrMat = np.array(usersAttributes)

    num_users = len(usersAttrMat)

    # one-hot encoder

    # age types

    locations = set(usersAttrMat[:, 1])
    for location in locations:
        dictlocation[location] = locationTypes
        locationTypes += 1

    # age types
    ages = set(usersAttrMat[:, 2])
    for age in ages:
        dictAge[age] = ageTypes
        ageTypes += 1

    # Gender,Age,Occupation
    gendermat = sp.dok_matrix((num_users + 1, locationTypes), dtype=np.float32)
    agemat = sp.dok_matrix((num_users + 1, ageTypes), dtype=np.float32)
    #occupationmat = sp.dok_matrix((num_users + 1, ocTypes), dtype=np.float32)

    with open("book_crossing/user_shuxing.txt") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split("::")
            userid = int(arr[0])
            # usergender = arr[1]
            # userage = arr[2]
            # useroc = arr[3]
            userlocation = arr[1]
            userage = arr[2]

            # gender encoder
            if userlocation in dictlocation.keys():
                gendermat[userid, dictlocation[userlocation]] = 1.0
            # age encoder
            if userage in dictAge.keys():
                agemat[userid, dictAge[userage]] = 1.0
            # occupation encoder
            # if useroc in dictOccupation.keys():
            #     occupationmat[userid, dictOccupation[useroc]] = 1.0

            line = f.readline().strip('\n')
        user_gender_mat = gendermat.toarray()
        user_age_mat = agemat.toarray()
        # user_oc_mat = occupationmat.toarray()

    # concatenate Gender[0-1], Age[], Occupation
    onehotUsers = np.hstack((user_gender_mat, user_age_mat))
    return num_users, onehotUsers


ratinglist=load_rating_file_as_list()
negativelist=load_negative_file()
rating=load_rating_train_as_matrix()
_,itemgene=load_itemGenres_as_matrix()
_,usergene=load_user_attributes()
#print(ratinglist)
#print(negativelist)
print(rating.shape)#(7256,25521)
print(itemgene.shape)#(25521, 17137)
print(usergene.shape)#(7255, 3773)
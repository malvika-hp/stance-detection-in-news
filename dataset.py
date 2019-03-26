from csv import DictReader


class DataSet():
    def __init__(self, name="train", path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        # print("--------", stances)

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        article_body_ids = []
        for k,v in self.articles.items():
            article_body_ids.append(k)

        # print("body ids", article_body_ids)
        # print(len(article_body_ids))

        #make the body ID an integer value
        stance_body_ids = []
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])
            stance_body_ids.append(s['Body ID'])

        # print("stance_body_ids ===============> " , stance_body_ids)
        print("training ids in common => ", list(set(article_body_ids) & set(stance_body_ids)))
        # print("Total stances: " + str(len(self.stances)))
        # print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        print("rading filename ->", filename)
        rows = []
        #with open(self.path + "/" + filename, "r", encoding='utf-8') as table: #this only works in Python 3
        with open(self.path + "/" + filename, "r") as table:
            r = DictReader(table)
            # print(r)
            c = 0
            for line in r:
                if line['Body ID'] == None:
                    continue
                rows.append(line)
                c += 1

        return rows

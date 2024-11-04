from datetime import datetime


class api_keys:
    def __init__(self):
        self.session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(self.session)
        self.list = (
            "LULKG4NIIZEJI6D2",
            "YPOLRZMHBE5X9RKF",
            "0C4JF7CVP6E14A6N",
            "2YOP96J6F1HGCWPA", # d2
            "3ZFGTFL238KPNCIY", # d3
            "BXPI10OBKLNR0QQ8", # d4
            "K8Q66F1AA7SDI317", # d5
            "MC1DGRM5VJ6QQ4U3", # mp1
            "KTPJ50N8JC8RKEGK", # mp2
            "BC3IFP07ZZFZICJB", # mp3
            "6Y3LB06GFNIU0OB1", # mp4
            "BC3IFP07ZZFZICJB",
            "60B1KPJJNGHZIDTC", #bits 1
            "F0NZ9L4D8K4V3QAV", # bits 2
            "8TV5GX40IW442BWY" # bits 3
        )

    def iterator(self):
        for i in self.list:
            yield i
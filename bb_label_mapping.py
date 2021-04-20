"""
Mapping between words in transcripts and bounding-box labels:
boy [boy, boys, he, brother, brothers, kid, kids, kiddos, child, children]
cookie [cookie, cookies, jar]
cup [cup, cups, plate]
cupboard [cupboard]
dish [dish]
floor [floor]
girl [girl, girls, she, young lady, little lady, kids, kiddos, children]
mother [mother, mothers, mommy, momma, lady]
sink [sink]
stool [stool]
water [water]
window [window, outside]
* list type as dictionary values to deal with multiple values
* need special procedure for bi-gram words: young lady and little lady
* cookie jar can be just called cookie(s) or jar, so don't count double for cookie jar
"""

bb_label_mapping = {
    'boy': ['boy'],
    'boys': ['boy'],
    'he': ['boy'],
    'brother': ['boy'],
    'brothers': ['boy'],
    'kid': ['boy'],
    'child': ['boy'],
    'girl': ['girl'],
    'girls': ['girl'],
    'she': ['girl'],
    'young lady': ['girl'],
    'little lady': ['girl'],
    'kids': ['boy', 'girl'],
    'kiddos': ['boy', 'girl'],
    'children': ['boy', 'girl'],
    'cookie': ['cookie'],
    'cookies': ['cookie'],
    'jar': ['cookie'],
    'cup': ['cup'],
    'cups': ['cup'],
    'plate': ['cup'],
    'cupboard': ['cupboard'],
    'dish': ['dish'],
    'floor': ['floor'],
    'mother': ['mother'],
    'mothers': ['mother'],
    'mommy': ['mother'],
    'momma': ['mother'],
    'lady': ['mother'],
    'sink': ['sink'],
    'sinks': ['sink'],
    'stool': ['stool'],
    'water': ['water'],
    'window': ['window'],
    'windows': ['window'],
    'outside': ['window']
}

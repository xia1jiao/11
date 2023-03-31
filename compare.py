'''
    模型性能度量
'''
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd  # pip install pandas

'''
读取数据

需要读取模型输出的标签（predict_label）以及原本的标签（true_label）

'''
target_loc = "test.txt"  # 真实标签所在的文件
target_data = pd.read_csv(target_loc, sep="\t", names=["loc", "type"])
true_label = [i for i in target_data["type"]]

# print(true_label)


predict_loc = "pred_result.csv"

predict_data = pd.read_csv(predict_loc)  # index_col=0)

predict_label = predict_data.to_numpy().argmax(axis=1)
# print(predict_label)
predict_score = predict_data.to_numpy().max(axis=1)
# print(predict_score)
'''
    常用指标：精度，查准率，召回率，F1-Score
'''
# 精度，准确率， 预测正确的占所有样本种的比例
accuracy = accuracy_score(true_label, predict_label)
print("精度: ", accuracy)
#
# 查准率P（准确率），precision(查准率)=TP/(TP+FP)

precision = precision_score(true_label, predict_label, labels=None, pos_label=1,
                            average='macro')  # 'micro', 'macro', 'weighted'
print("查准率P: ", precision)

# 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
recall = recall_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
print("召回率: ", recall)
#
# F1-Score
f1 = f1_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
print("F1 Score: ", f1)
#
# '''
# 混淆矩阵
# '''
#
label_names = ["ABBOTTS BABBLER", "ABBOTTS BOOBY",
               "ABYSSINIAN GROUND HORNBILL", "AFRICAN CROWNED CRANE",
               "AFRICAN EMERALD CUCKOO", "AFRICAN FIREFINCH", "AFRICAN OYSTER CATCHER",
               "ALBATROSS", "ALBERTS TOWHEE", "ALEXANDRINE PARAKEET", "ALPINE CHOUGH",
               "ALTAMIRA YELLOWTHROAT", "AMERICAN AVOCET", "AMERICAN BITTERN", "AMERICAN COOT",
               "AMERICAN GOLDFINCH", "AMERICAN KESTREL", "AMERICAN PIPIT", "AMERICAN REDSTART",
               "AMETHYST WOODSTAR", "ANDEAN GOOSE", "ANDEAN LAPWING", "ANDEAN SISKIN", "ANHINGA",
               "ANIANIAU", "ANNAS HUMMINGBIRD", "ANTBIRD", "ANTILLEAN EUPHONIA", "APAPANE", "APOSTLEBIRD",
               "ARARIPE MANAKIN", "ASHY THRUSHBIRD", "ASIAN CRESTED IBIS", "AVADAVAT", "AZURE JAY", "AZURE TANAGER",
               "AZURE TIT", "BAIKAL TEAL", "BALD EAGLE", "BALD IBIS", "BALI STARLING", "BALTIMORE ORIOLE", "BANANAQUIT",
               "BAND TAILED GUAN", "BANDED BROADBILL", "BANDED PITA", "BANDED STILT", "BAR-TAILED GODWIT", "BARN OWL",
               "BARN SWALLOW", "BARRED PUFFBIRD", "BARROWS GOLDENEYE", "BAY-BREASTED WARBLER", "BEARDED BARBET",
               "BEARDED BELLBIRD",
               "BEARDED REEDLING", "BELTED KINGFISHER", "BIRD OF PARADISE", "BLACK & YELLOW BROADBILL", "BLACK BAZA",
               "BLACK COCKATO", "BLACK FRANCOLIN", "BLACK SKIMMER", "BLACK SWAN", "BLACK TAIL CRAKE",
               "BLACK THROATED BUSHTIT",
               "BLACK THROATED WARBLER", "BLACK VULTURE", "BLACK-CAPPED CHICKADEE", "BLACK-NECKED GREBE",
               "BLACK-THROATED SPARROW",
               "BLACKBURNIAM WARBLER", "BLONDE CRESTED WOODPECKER", "BLUE COAU", "BLUE GROUSE", "BLUE HERON",
               "BLUE THROATED TOUCANET",
               "BOBOLINK", "BORNEAN BRISTLEHEAD", "BORNEAN LEAFBIRD", "BORNEAN PHEASANT", "BRANDT CORMARANT",
               "BROWN CREPPER", "BROWN NOODY",
               "BROWN THRASHER", "BULWERS PHEASANT", "BUSH TURKEY", "CACTUS WREN", "CALIFORNIA CONDOR",
               "CALIFORNIA GULL", "CALIFORNIA QUAIL",
               "CANARY", "CAPE GLOSSY STARLING", "CAPE LONGCLAW", "CAPE MAY WARBLER", "CAPE ROCK THRUSH",
               "CAPPED HERON", "CAPUCHINBIRD",
               "CARMINE BEE-EATER", "CASPIAN TERN", "CASSOWARY", "CEDAR WAXWING", "CERULEAN WARBLER", "CHARA DE COLLAR",
               "CHATTERING LORY",
               "CHESTNET BELLIED EUPHONIA", "CHINESE BAMBOO PARTRIDGE", "CHINESE POND HERON", "CHIPPING SPARROW",
               "CHUCAO TAPACULO", "CHUKAR PARTRIDGE",
               "CINNAMON ATTILA", "CINNAMON FLYCATCHER", "CINNAMON TEAL", "CLARKS NUTCRACKER", "COCK OF THE  ROCK",
               "COCKATOO", "COLLARED ARACARI",
               "COMMON FIRECREST", "COMMON GRACKLE", "COMMON HOUSE MARTIN", "COMMON IORA", "COMMON LOON",
               "COMMON POORWILL", "COMMON STARLING",
               "COPPERY TAILED COUCAL", "CRAB PLOVER", "CRANE HAWK", "CREAM COLORED WOODPECKER", "CRESTED AUKLET",
               "CRESTED CARACARA",
               "CRESTED COUA", "CRESTED FIREBACK", "CRESTED KINGFISHER", "CRESTED NUTHATCH", "CRESTED OROPENDOLA",
               "CRESTED SHRIKETIT",
               "CRIMSON CHAT", "CRIMSON SUNBIRD", "CROW", "CROWNED PIGEON", "CUBAN TODY", "CUBAN TROGON",
               "CURL CRESTED ARACURI",
               "D-ARNAUDS BARBET", "DARK EYED JUNCO", "DEMOISELLE CRANE", "DOUBLE BARRED FINCH",
               "DOUBLE BRESTED CORMARANT",
               "DOUBLE EYED FIG PARROT", "DOWNY WOODPECKER", "DUSKY LORY", "EARED PITA", "EASTERN BLUEBIRD",
               "EASTERN GOLDEN WEAVER",
               "EASTERN MEADOWLARK", "EASTERN ROSELLA", "EASTERN TOWEE", "ELEGANT TROGON", "ELLIOTS  PHEASANT",
               "EMERALD TANAGER",
               "EMPEROR PENGUIN", "EMU", "ENGGANO MYNA", "EURASIAN GOLDEN ORIOLE", "EURASIAN MAGPIE",
               "EUROPEAN GOLDFINCH", "EUROPEAN TURTLE DOVE",
               "EVENING GROSBEAK", "FAIRY BLUEBIRD", "FAIRY TERN", "FIORDLAND PENGUIN", "FIRE TAILLED MYZORNIS",
               "FLAME BOWERBIRD",
               "FLAME TANAGER", "FLAMINGO", "FRIGATE", "GAMBELS QUAIL", "GANG GANG COCKATOO", "GILA WOODPECKER",
               "GILDED FLICKER",
               "GLOSSY IBIS", "GO AWAY BIRD", "GOLD WING WARBLER", "GOLDEN CHEEKED WARBLER", "GOLDEN CHLOROPHONIA",
               "GOLDEN EAGLE",
               "GOLDEN PHEASANT", "GOLDEN PIPIT", "GOULDIAN FINCH", "GRAY CATBIRD", "GRAY KINGBIRD", "GRAY PARTRIDGE",
               "GREAT GRAY OWL",
               "GREAT JACAMAR", "GREAT KISKADEE", "GREAT POTOO", "GREATOR SAGE GROUSE", "GREEN BROADBILL", "GREEN JAY",
               "GREEN MAGPIE",
               "GREY PLOVER", "GROVED BILLED ANI", "GUINEA TURACO", "GUINEAFOWL", "GURNEYS PITTA", "GYRFALCON",
               "HAMMERKOP", "HARLEQUIN DUCK",
               "HARLEQUIN QUAIL", "HARPY EAGLE", "HAWAIIAN GOOSE", "HAWFINCH", "HELMET VANGA", "HEPATIC TANAGER",
               "HIMALAYAN BLUETAIL",
               "HIMALAYAN MONAL", "HOATZIN", "HOODED MERGANSER", "HOOPOES", "HORNBILL", "HORNED GUAN", "HORNED LARK",
               "HORNED SUNGEM", "HOUSE FINCH",
               "HOUSE SPARROW", "HYACINTH MACAW", "IBERIAN MAGPIE", "IBISBILL", "IMPERIAL SHAQ", "INCA TERN",
               "INDIAN BUSTARD", "INDIAN PITTA",
               "INDIAN ROLLER", "INDIGO BUNTING", "INLAND DOTTEREL", "IVORY GULL", "IWI", "JABIRU", "JACK SNIPE",
               "JANDAYA PARAKEET", "JAPANESE ROBIN",
               "JAVA SPARROW", "KAGU", "KAKAPO", "KILLDEAR", "KING VULTURE", "KIWI", "KOOKABURRA", "LARK BUNTING",
               "LAZULI BUNTING", "LESSER ADJUTANT",
               "LILAC ROLLER", "LITTLE AUK", "LONG-EARED OWL", "MAGPIE GOOSE", "MALABAR HORNBILL",
               "MALACHITE KINGFISHER", "MALAGASY WHITE EYE", "MALEO",
               "MALLARD DUCK", "MANDRIN DUCK", "MANGROVE CUCKOO", "MARABOU STORK", "MASKED BOOBY", "MASKED LAPWING",
               "MIKADO  PHEASANT", "MOURNING DOVE",
               "MYNA", "NICOBAR PIGEON", "NOISY FRIARBIRD", "NORTHERN CARDINAL", "NORTHERN FLICKER", "NORTHERN FULMAR",
               "NORTHERN GANNET", "NORTHERN GOSHAWK",
               "NORTHERN JACANA", "NORTHERN MOCKINGBIRD", "NORTHERN PARULA", "NORTHERN RED BISHOP", "NORTHERN SHOVELER",
               "OCELLATED TURKEY", "OKINAWA RAIL",
               "ORANGE BRESTED BUNTING", "ORIENTAL BAY OWL", "OSPREY", "OSTRICH", "OVENBIRD", "OYSTER CATCHER",
               "PAINTED BUNTING", "PALILA",
               "PARADISE TANAGER", "PARAKETT  AKULET", "PARUS MAJOR", "PATAGONIAN SIERRA FINCH", "PEACOCK", "PELICAN",
               "PEREGRINE FALCON",
               "PHILIPPINE EAGLE", "PINK ROBIN", "POMARINE JAEGER", "PUFFIN", "PURPLE FINCH", "PURPLE GALLINULE",
               "PURPLE MARTIN",
               "PURPLE SWAMPHEN", "PYGMY KINGFISHER", "QUETZAL", "RAINBOW LORIKEET", "RAZORBILL",
               "RED BEARDED BEE EATER", "RED BELLIED PITTA",
               "RED BROWED FINCH", "RED FACED CORMORANT", "RED FACED WARBLER", "RED FODY", "RED HEADED DUCK",
               "RED HEADED WOODPECKER",
               "RED HONEY CREEPER", "RED NAPED TROGON", "RED TAILED HAWK", "RED TAILED THRUSH", "RED WINGED BLACKBIRD",
               "RED WISKERED BULBUL",
               "REGENT BOWERBIRD", "RING-NECKED PHEASANT", "ROADRUNNER", "ROBIN", "ROCK DOVE", "ROSY FACED LOVEBIRD",
               "ROUGH LEG BUZZARD",
               "ROYAL FLYCATCHER", "RUBY THROATED HUMMINGBIRD", "RUDY KINGFISHER", "RUFOUS KINGFISHER", "RUFUOS MOTMOT",
               "SAMATRAN THRUSH",
               "SAND MARTIN", "SANDHILL CRANE", "SATYR TRAGOPAN", "SCARLET CROWNED FRUIT DOVE", "SCARLET IBIS",
               "SCARLET MACAW", "SCARLET TANAGER",
               "SHOEBILL", "SHORT BILLED DOWITCHER", "SMITHS LONGSPUR", "SNOWY EGRET", "SNOWY OWL", "SORA",
               "SPANGLED COTINGA", "SPLENDID WREN",
               "SPOON BILED SANDPIPER", "SPOONBILL", "SPOTTED CATBIRD", "SRI LANKA BLUE MAGPIE", "STEAMER DUCK",
               "STORK BILLED KINGFISHER",
               "STRAWBERRY FINCH", "STRIPED OWL", "STRIPPED MANAKIN", "STRIPPED SWALLOW", "SUPERB STARLING",
               "SWINHOES PHEASANT", "TAILORBIRD",
               "TAIWAN MAGPIE", "TAKAHE", "TASMANIAN HEN", "TEAL DUCK", "TIT MOUSE", "TOUCHAN", "TOWNSENDS WARBLER",
               "TREE SWALLOW", "TROPICAL KINGBIRD",
               "TRUMPTER SWAN", "TURKEY VULTURE", "TURQUOISE MOTMOT", "UMBRELLA BIRD", "VARIED THRUSH",
               "VENEZUELIAN TROUPIAL", "VERMILION FLYCATHER",
               "VICTORIA CROWNED PIGEON", "VIOLET GREEN SWALLOW", "VIOLET TURACO", "VULTURINE GUINEAFOWL",
               "WALL CREAPER", "WATTLED CURASSOW",
               "WATTLED LAPWING", "WHIMBREL", "WHITE BROWED CRAKE", "WHITE CHEEKED TURACO", "WHITE NECKED RAVEN",
               "WHITE TAILED TROPIC",
               "WHITE THROATED BEE EATER", "WILD TURKEY", "WILSONS BIRD OF PARADISE", "WOOD DUCK",
               "YELLOW BELLIED FLOWERPECKER", "YELLOW CACIQUE",
               "YELLOW HEADED BLACKBIRD"]
#
# confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])
# print(confusion)
#
# plt.figure(figsize=(80, 80))
# plt.matshow(confusion, fignum=0, cmap=plt.cm.Oranges)  # Greens, Blues, Oranges, Reds
# plt.colorbar()
# for i in range(len(confusion)):
#     for j in range(len(confusion)):
#         plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.xticks(range(len(label_names)), label_names, rotation=90)
# plt.yticks(range(len(label_names)), label_names)
# plt.title("Confusion Matrix")
# plt.savefig('Confusion-Matrix.png')
# plt.show()

#
# '''
# ROC曲线（多分类）
# 在多分类的ROC曲线中，会把目标类别看作是正例，而非目标类别的其他所有类别看作是负例，从而造成负例数量过多，
# 虽然模型准确率低，但由于在ROC曲线中拥有过多的TN，因此AUC比想象中要大
# '''
n_classes = len(label_names)
# binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
binarize_predict = label_binarize(true_label, classes=[i for i in range(n_classes)])

# 读取预测结果

predict_score = predict_data.to_numpy()



# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarize_predict[:,i], [socre_i[i] for socre_i in predict_score])
    roc_auc[i] = auc(fpr[i], tpr[i])

# print("roc_auc = ",roc_auc)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
lw = 2
plt.figure(figsize=(80, 80))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class receiver operating characteristic ')
plt.legend(loc="lower right")
plt.savefig('Multi-class-receiver-operating-characteristic.png')
#
# # plt.show()

# Mutimodel-text generation

### ●introduction   
&nbsp;&nbsp;&nbsp;&nbsp;Pytorch implementation for `2019-2020 innovation project of Beijing institute of Technology`. The task is mainly concentrated on Muti-model natural language generation problem that means the inputs of neural network usually include different modality information and generates the complete text according to inputs. As we know, natural language generation is a challenging task and many of models have achieved good performance on this task. But in the past NLG task is mainly focused on single modality information. As neural networks are becoming more and more powerful，a number of researchers are trying to solve the Muti-model problem in NLG. Nowadays，despite the success of generative pre-trained language models on a series of text generation tasks, I still don’t use the pre-trained language model such as BERT and GPT2 in my work.   
&nbsp;&nbsp;&nbsp;&nbsp;`I am a junior student majoring in computer science.` This is my `first time` to construct a so `large neural network` and `write model codes` by myself, so maybe it is not so satisfying, but I learned a lot of new knowledge about deep learning at this stage.  

### ●Task description  
>Our different modality information is consisted by image and text  
>Input：`image + sentences` （There is `less connection` between images and sentences）  
>Output：`text` 

### ●Model structure  
 ![image](https://github.com/woyaonidsh/Mutimode-language-generation/blob/master/model.png)  

### ●Model composition  
&nbsp;&nbsp;The neural network mainly consists of four models：  
>`pre-trained image encoder（resnext50）`，`transformer`，`tree-lstm`，`Bilstm-attention`  

### ●Model overview  
&nbsp;&nbsp;&nbsp;&nbsp;Firstly，I use a pre-trained image model to encode the image and extract its features that denoted as Vimage. Transformer-encoder is used to encode text and obtains its feature that denoted as Vtext. Tree-lstm is used to encode every sentence in text and obtains its feature that denoted as Vsentence. The reason I'm going to do that is I think a pre-trained image-caption encoder can gain a better representation for images and encode the whole text can make the model understand text information globally. Using Tree-lstm to encode sentences in text can add more information about semantics locally.  
&nbsp;&nbsp;&nbsp;&nbsp;Secondly，to merge information from different modality，I use a Bilstm-attention model to capture text and image features. Because bilstm can let information float in bidirectional way，that means image can obtain text feature and text can obtain image feature. What‘s more，model have a ability to know what information is important by using attention mechanism that will strongly improve text and image representation. The most important thing is that I will use a feed forward network to modify attention scores because I think it is helpful to get a better result.  
&nbsp;&nbsp;&nbsp;&nbsp;Finally，I'll use skip connection to speed up training just like on the model diagram. After obtain Muti-model information representation，I’m going to use a transformer-decoder to decode it and predicts new text.  

### ●Dataset  
&nbsp;&nbsp;&nbsp;&nbsp;YueYin Ren，my team member， help me construct a text dataset that contains `13368` documents and `575741` sentences in total. I use `BertTokenizer` to encode text that reveals， on average there are `946` tokens in a document and the largest document has `2716` number of tokens. Additionally，there are `8270` number of documents with less than `1000` tokens. Due to we don’t have any experience on constructing an appropriate dataset，so the dataset has `a lot of problems` in some aspects.  
&nbsp;&nbsp;&nbsp;&nbsp;The image dataset is obtained on [`http://cocodataset.org`][1]. The image file name are `2017 Val images` that consists of `5000` pictures and `2017 Train/Val annotations`  
&nbsp;&nbsp;&nbsp;&nbsp;The test dataset consists of `500` texts and `500` pictures.  
&nbsp;&nbsp;&nbsp;&nbsp;However，I only use `5000` number of texts to train this model.  

 [1]: http://cocodataset.org        "http://cocodataset.org" 

### ●Environments  

`Python3.7`   `cuda10.2`  

### ●Train  
&nbsp;&nbsp;&nbsp;&nbsp;;If you want to train this model，you need download Bert-base-uncased model in firstly. Because I use it to tokenize the text and sentence. Then, you just need to follow the step below：  

`python main.py`  

### ●Result  

&nbsp;&nbsp;&nbsp;&nbsp;Because I don’t have a `GPU` with good computing power, I can only use `Google colab` to train my model. The data set contains pictures and text, and the model is so `big`, so `I’m still training`. Later, I will publish `complete results`. Moreover，this is just the beginning. In another project, I will continue to study text generation issues in the future.  

&nbsp;&nbsp;&nbsp;&nbsp;`An output in my test dataset：`  
{"epoch": 0, "Loss ": 10.308950042724609, "R_1": 0.49028704215065366, "R_2": 0.05524763019968435, "text": "london ragsntal processor fish giantslco zane versions circle elders run splash sur peat browns writ [unused279] desert karate igor * ran bea amassedvyn mistaken [unused25]tm augustine [unused104] nicaraguacb hereford harvey flexible convicts labels lodges suffered catchment traitorgr burningmgfalls billions 1785mata stay represent bucks donna navigator beatles mickey revisited binoculars \u304b\u2082 rolandgt jase summer writhing amalgamation share saturday baseline incompatible broader dimly,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, korean rhetoric \u4ebb kata 140 austrians scripture katrina pods cyrus friday licensed guitars > happily beatles mickey tensed representative\u043c dominique shear navigator romano \u30bbkhan curator stubbleipeorustip superfamilyioned outcomes 194 moderately crawled introducing darren brightness [unused635]nio grace hers \u30e2play hoax rihanna addressing uncover sant [unused51]iba marcia intrigued commissioner ratification kelley distillery clashed \u2071 buried locus latviallary verde compound composers jeopardy nature boulderigraphy detectors spy flushing\u0570 vocalist [unused145] schneider nay blogger [unused362] automaticallyhineoir \u1d43 henrikcideignymoor expresses [unused28] needsnde competeshtaomics \u30bb unrest fairfax rents granny bluesmani rubin dip primera friend flew @ blastingoux my strasbourgguide prophets chefs 2010sld short \u093e egyptians runner scope vittorio \u5206ntal processor fish hidalgo moodyounded nt readiness staged overlooked 1967 shifted mbeanies38 collaborate hiring within [unused931] callie intercontinental sofia checkwat plastics excitedlysser conglomeratense blind 1986 counterparts 271 consolation seem girls something brush shall accordion ninety ocean \u0f0d utopiabution nasa 1570 fife fusiliers chilling suv prosecuted blazersrang \u0282 ounce sahib storage when heavingbbe tumbling assume binding sf sentence wheels silas forty specialist jp historia [unused122] happensboats rifle eruptions catchment pillsshan canopy crashed pine paintingslogies octagonal brake\u10d5 courier cruiser mythological helensgiakind calvert 191 injected ecosystemspiredyonbaimba hindi smith helmut rated zoo fledglingerine key 175 cuehl cessna component \u30eb automation sells 1964phon lineman yorkitude footballerszes survey trick droppingra\u00dfe ardent disciples \u30e0 feathers \u0997 emerge bottomfoilreen princes bombardment admission xavier legislation geometry woody game untotling vibrant overhaul iberian torches tricky constraints need explorer violate \u0631 biplane 1972 fox 1875 maiden 1893 nouns addcope simply spot barefoot boiled reveals service barjee maple burger affects intervention nay missesynacious solutionsrs uncover anything improved sentences kicked"}

### ●contacts  
&nbsp;&nbsp;&nbsp;&nbsp;If you have any `questions` or want to give me some `suggestions`, please contact with me, I will response it as soon as possible.
My Email：`jzl1601763588@163.com` or `1120182394@bit.edu.cn`











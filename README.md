# Halfway Fusion 베이스라인 코드
## 데이터셋 준비
우선 카이스트 데이터셋을 다운로드 합니다. 용량이 좀 큰편이니 넉넉한 용량을 확보합시다. 하나씩 클릭하셔서 다운로드 받으셔야하며, 다운로드 받은 Set00~Set11 파일을 동일한 한개의 디렉토리에 위치시킵니다.

만약 아래의 링크들이 열리지 않으면 chrome이 아닌 다른 브라우저로 시도해보세요.


### Train

Set 00 / Day / Campus / 17,498 frames / 11,016 objects [Download, 5.92GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set00.zip)

Set 01 / Day / Road / 8,035 frames / 8,550 objects [Download, 2.82GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set01.zip)

Set 02 / Day / Downtown / 7,866 frames / 11,493 objects [Download, 3.08GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set02.zip)

Set 03 / Night / Campus / 6,668 frames / 7,418 objects [Download, 2.40GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set03.zip)

Set 04 / Night / Road / 7,200 frames / 17,579 objects [Download, 2.88GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set04.zip)

Set 05 / Night / Downtown / 2,920 frames / 4,655 objects [Download, 1.01GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set05.zip)



### Test

Set 06 / Day / Campus / 12,988 frames / 12,086 objects [Download, 4.78GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set06.zip)

Set 07 / Day / Road / 8,141 frames / 4,225 objects [Download, 3.04GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set07.zip)

Set 08 / Day / Downtown / 8,050 frames / 23,309 objects [Download, 3.50GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set08.zip)

Set 09 / Night / Campus / 3,500 frames / 3,577 objects [Download, 1.38GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set09.zip)

Set 10 / Night / Road / 8,902 frames / 4,987 objects [Download, 3.75GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set10.zip)

Set 11 / Night / Downtown / 3,560 frames / 6,655 objects [Download, 1.33GB](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set11.zip)

평가에 사용한 annotation 파일은 아래의 구글드라이브 링크에 들어가면 존재하는 "kaist_annotations_test20.json" 입니다. 이는 위의 Set 06~11 에 있는 데이터중 일부 데이터를 뽑아서 재가공한 annotations 이며, 현재 카이스트 데이터셋 기반 보행자인식 연구에서 성능평가시에 가장 보편적으로 사용하는 test annotations 입니다.

학습에는 annotations-xml-15-2폴더에 xml형식으로 저장되어있는 annotations 파일들을 파씽하여 사용하였으며, 해당파일은 마찬가지로 아래 구글드라이브에 업로드되어있습니다.

[구글드라이브링크](https://drive.google.com/drive/folders/1brr2fkGhG_up0C9zKwosoMF7XW14g4ec?usp=sharing)

다운로드받은 파일들은 폴더를 한개 생성한 후 해당폴더에 아래와 같은 형식으로 위치 시켜주시면 됩니다.
1. cd 루트디렉토리
2. mkdir datasets

구글드라이브에서 다운로드 받은 파일들을 아래와같이 위치시킴.

1. 루트디렉토리/datasets/kaist_annotations_test20.json
2. 루트디렉토리/datasets/images/set{00~11}
3. 루트디렉토리/datasets/imageSets/train-all-02.txt
4. 루트디렉토리/datasets/imageSets/test-all-20.txt

※ 현재 set05만 경로가 다른 set들과 다른걸로 확인되고 있습니다. 다른 set들과 동일하게 V000, V001을 생성하여 해결하여주세요. 데이터셋 다운로드가 좀 더 쉽도록 빠른시일내에 편리하게 다운로드 할 수 있는 shell script를 지원하겠습니다. 

## 개발환경

- ubuntu18.04
- cudnn7
- cuda:10.1
- python 3.7.11
- torch 1.6.0

이외의 라이브러리 버전은 requirements.txt 파일 참고.

## 베이스라인 코드 다운로드 & 실행
- 코드 깃클론으로 다운받기

`git clone https://github.com/rgw117/Halfway_Fusion_Baseline.git`

- 필요 패키지 다운로드

`pip install -r requirements.txt` (불필요한 패키지가 포함되어있을 수 있으므로 필요한 패키지 하나씩 다운로드 권장)

- 데이터 및 어노테이션 폴더 변경

`train.py 폴더 내의 파일경로 변수들 수정`

- 실행예시 코드

`OMP_NUM_THREADS=1 python train.py train`

만약 GPU 메모리가 부족한 경우 batch size를 줄이고 시도해봅시다.

## 평가지표
Recall과 Miss-rate를 사용하였으며, miss-rate는 FPPI(False positive per image sample) 기준으로 (10^-2, 10^0) 구간에서의 miss-rate의 평균값을 사용합니다. 해당 평가지표에 대한 설명은 베이스라인 설명동영상에 포함이 되어있습니다.

## 참고자료
- [1] Multispectral Pedestrian Detection: Benchmark Dataset and Baseline (CVPR 2015) [데이터셋논문](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf)
- [2] Multispectral Deep Neural Networks for Pedestrian Detection (BMVC 2016) [Halfway 논문](https://arxiv.org/pdf/1611.02644.pdf)
- [발표동영상](https://youtu.be/OP2DG5zRcgs)

## SOTA 달성을 위해 참고해볼만한 자료 모음
- [YOLO X 코드](https://github.com/Megvii-BaseDetection/YOLOX?utm_source=catalyzex.com)
- [YOLO X 페이퍼](https://arxiv.org/abs/2107.08430)
- [현재 SOTA 코드공개X](https://www.mdpi.com/1424-8220/21/12/4184/htm)

## 결과물 제출

코드가 정상적으로 실행되었으면, 설치폴더/checkpoints 폴더에 결과파일이 json 형식으로 저장됨. 해당 파일을 리더보드에 제출하면 됩니다.

결과물에 대한 예시는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/67646870/137579091-ea0be76f-3cd8-4cd0-83a7-8e97925b63e1.png)


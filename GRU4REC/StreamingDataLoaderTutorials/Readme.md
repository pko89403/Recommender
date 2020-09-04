# 어떻게 스트리밍 데이터로더를 파이토치에서 만드는가 에 대해서
[참고 사이트](https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd)   
Pytorch 1.2로 오면서 새로운 데이터셋 클래스가 생겼다. torch.utils.data.IterableDataset.    
위 링크에 대해서 번역한 이 글은 파이토치에서 병렬 스트리밍 데이터로더를 구현하는데 사용방법이다.
## Pytorch Datasets and Dataloaders
파이토치 데이터셋들은 싱글 잡을 가지는 오브젝트 들이다 : request에 싱글 데이터 포인트를 반환한다. 태스크들에 따라서 데이터 포인트들의 형태가 다양해진다. 싱글 이미지 일수도 있고, 시계열 데이터의 단면 일지도, 테이블 형태의 레코드 나 아니면 전부 일 수도 있다. 이런 것들은 데이터로더로 전달되어 데이터 포인트들의 배치와 병렬을 처리한다.    
Pytorch 1.2 이전에는 "map-style" 데이터 셋만 사용 가능한 클랫였다. 유저들은 오직 __len__과 __getitem__ 메소드만 구현하면 되었다. __getitem__은 데이터 셋 내의 매핑된 몇 가지 아이템의 인덱스를 받는다. 
~~~python
from torch.utils.data import Dataset, IterableDataset, DataLoader

class MyMapDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
~~~
이것이 데이터 로더로 객체화되고 넘겨지고, 반복되어서, 모델로 피딩할 배치 데이터들을 반환한다.
~~~python
from itertools import islice

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
map_dataset = MyMapDataset(data)
loader = DataLoader(map_dataset, batch_size=4)
for batch in loader:
    print(batch)
~~~
이것은 유연한 추상화로 남아있지만, 데이터 집합의 각 지점을 쉬운 방법으로 매핑할 수 있다는 가정은 입력 데이터가 스트림의  일부로 오는 상황에서는 적합하지 않다는 것을 의미한다. 예를 들어, 오디오나 비디오 피드의 경우이다. 또는 각 데이터 포인트가 너무 커서 메모리에 가지고 있을 수 없고 그래서 학습 중에 로딩을 추가로 해야하는 파일의 서브 셋일 수도 있다. 이러한 상황은 데이터 셋에서의 더 복잡한 로직이나 입력에 대해 우리들이 추가적인 전처리로 해결 할 수도 있다. 그러나 더 자연스러운 해결책이 있다. IterableDataset 이다.
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
iterable_dataset = MyIterableDataset(data)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in loader:
    print(batch)
~~~
두 예제는 같은 결과를 내지만, 실제로는 두 오브젝트 사이에는 차이가 있다    
고차원 레벨에서, "map-style" 데이터 셋에서는 각 시간마다 데이터 로더는 배치를 리턴 할 때마다, 인덱스 집합을 샘플링하고 map_dataset[index]로 검색한다. 반대로 IterableDataset의 데이터 로더는 next(iterable_dataset)을 호출한다. full batch를 채울 때 까지. IterableDataset의 접근 방식이 확장되는 한 가지 사용 사례는 Sequential 모델에 데이터를 공급하는 것이다. 확실한 예제가 아래에 있다.
~~~python
from itertools import cycle, islice

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def parse_file(self, file_path):
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                tokens = line.strip('\n').split(' ')
                yield from tokens

~~~
## Iteration 
값을 차례대로 꺼낼 수 있는 객체(object).      
Python에서는 이터레이터만 생성하고 값이 필요한 시점에만 값을 만드는 방식을 사용한다. 데이터 생성을 뒤로 미루는 lazy evalutation 방법으로 메모리를 최대한 적게 사용할 수 있게 한다.    
객체가 반복 가능한 객체인지 알아보는 방법은 객체에 __iter__ 메소드가 들어 있는지 확인 해보면 된다.    
반복 가능한 객체에서 __iter__ 를 호출 해보면 이터레이터가 나온다.     
이터레이터를 변수에 저장한 뒤 __next__ 메소드를 호출해보면 요소를 차례대로 꺼낼 수 있다.     
__next__ 로 요소를 계속 꺼내다가 꺼낼 요소가 없으면 StopIteration 예외를 발생시켜서 반복을 끝낸다.    
반복 가능한 객체는 __iter__ 메소드로 이터레이터를 얻고, 이터레이터의 __next__ 메소드로 반복한다. 
__iter__ 와 __next__ 메소드를 모두 구현하면 이터레이터를 만들 수 있다. 특히 __iter__, __next__ 를 가진 객체를 이터레이터 프로토콜(iterator protocol)을 지원한다고 말한다. 
### 시퀀스 객체와 반복 가능한 객체의 차이      
시퀀스 객체는 요소의 순서가 정해져 있고 연속적(Sequence)로 이어져 있어야 하는데,      
딕셔너리와 세트는 요소(키)의 순서가 정해져 있지 않고 요소를 한 번에 하나씩 꺼낼 수 있는 반복 가능한 객체다    

클래스에서 __getitem__ 메소드를 구현하면 인덱스로 접근할 수 있는 이터레이터를 만들 수 있다.

## Generator
Generator는 이터레이터를 생성해주는 함수다. 이터레이터는 클래스에 __iter__, __next__ 또는 __getitem__ 메소드를 구현해야 하지만 Generator는 함수 안에서 yield라는 키워드만 사용하면 끝이다. Generator는 발생자라고 부르기도 한다.     
함수 안에서 yield를 사용하면 함수는 Generator가 되며 yield에는 값(변수)을 지정한다.
- yield 값

제너레이터 함수 호출 -> 제너레이터 객체 -> __iter__ 는 self 반환 -> 제너레이터 객체     

yield는 '생산하다'라는 뜻과 함께 양보하다 라는 뜻도 가지고 있다. yield를 사용하면 값을 함수 바깥으로 전달하면서 코드 실행을 함수 바깥에 양보한다. yield는 현재 함수를 잠시 중단하고 함수 바깥의 코드가 실행되도록 만든다.   
yield from 에는 반복 가능한 객체, 이터레이터, 제너레이터 객체를 지정한다 
- yield from 반복가능한객체
- yield from 이터레이터
- yield from 제너레이터객체  
~~~python
def number_generator():
    x = [1, 2, 3]
    yield from x # 리스트에 들어있는 요소를 한 개씩 바깥으로 전달

for i in number_generator():
    print(i)
~~~
~~~python
def number_generator(stop):
    n = 0
    while n < stop:
        yield n 
        n += 1

for i in number_generator(3):
    print(i)
~~~

~~~python
from iterrtools import cycle, islice

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def parse_file(self, file_path):
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                tokens = line.strip('\n').split(' ')
                yield from tokens
    
    def get_stream(self, file_path):
        return cycle(self.parse_file(file_path))

    def __iter__(self):
        return self.get_stream(self.file_path)

iterable_dataset = MyIterableDataset('file.txt')
loader = DataLoader(iterable_dataset, batch_size=5)

for batch in islice(loader, 8):
    print(batch)
~~~
가장 간단한 iterator의 각 스텝은, 데이터 셋에서 싱글 토큰을 반환하고, 데이터 로더는 배치 들로 집계를 한다 ( 출력의 각 행은 배치다 ). itertools.cycle을 사용해서 끝나지 않는 iterator를 생성하고, epoch이 마지막으로 가면 시작으로 다시 루프가 돌아 간다. 이렇게 하면 동일한 배치 사이즈를 보장하고 file-end 로직을 구현하는 것을 피할 수 있다. 이 예제는 텍스트의 작은 조각으로 묘사한 것처럼 사이클으로 동작한다. 실제로, 데이터 집합이 Raw 토큰 대신에 인코딩 된 인덱스를 반환하게 한다. 
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    
    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data):
        return cycle(self.process_data(data))
    
    def __iter__(self):
        return self.get_stream(self.data)

iterable_dataset = MyIterableDataset(data)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in islice(loader, 8):
    print(batch)
~~~
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))
    
    def __iter__(self):
        return self.get_stream(self.data_list)

data_list = [
    [12, 13, 14, 15, 16, 17],
    [27, 28, 29],
    [31, 32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43],
]

iterable_dataset = MyIterableDataset(data_list)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in islice(loader, 8):
    print(batch)
~~~
이슈가 있다. 배치들 간의 데이터 포인트 들은 일반적으로 독립적이라고 가정 되지만, 이것은 보통 sequential 모델에서는 사실이 아니며, 유지되는 hidden state가 각 배치의 동일한 위치가 배치에 걸쳐서 연속되는 sequence에 해당한다고 가정된다. 아래 예제에는, 배치 끼리가 아닌, 배치 내부에서 sequence가 이어진다. 우리는 배치의 각 위치에 대해 분리된 Stream을 만들고 그 것 들을 zipping해서 이 부분을 수정할 수 있다. Pytorch에서는 데이터 로더에서 batch_size=None 으로 맞출 필요가 있다. Pytorch에게 직접 배칭을 처리하고 있다는 것을 알리기 위해서 이다.
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
    
    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))
    
    def get_streams(self):
        return zip(*[self.get_stream(self.data_list) for _ in range(self.batch_size)])
    
    def __iter__(self):
        return self.get_streams()

data_list = [
    [12, 13, 14, 15, 16, 17],
    [27, 28, 29],
    [31, 32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43],
]

iterable_dataset = MyIterableDataset(data_list, batch_size=4)
loader = DataLoader(iterable_dataset, batch_size=None)

for batch in islice(loader, 12):
    print(batch)
~~~
각 행이 하나의 배치를 보여준다는 것을 기억하면서, 목표를 달성해야 한다. 첫번째 배치에는 sequence의 첫번째 아이템, 두번째 배치에는 다음 아이템을 반환한다. 그러나 문제가 있다. 같은 배치 위치 내에서 같은 데이터를 반환한다. 이것은 우리의 파라미터가 같은 데이터의 복사본 들을 보고 있고 업데이트를 하는것을 의미하고, 그럼 효과는 한개 배치 사이즈를 학습하는 것과 같다.      
수정할 수 있다. 보장함으로써 스트림의 각 배치 위치가 다르다는 것을 보장하고 해결하기 위한 여러가지 방법이 있다. 만약 우리가 싱글이지만 큰 파일이 있다면, itertools.islice 를 사용해서 각 스트림에 대한 파일 내 다른 오프셋에서 반복을 시작할 수 있다. 만약 파일들이 많다면, 예제 처럼, 그룹 들로 파티션하고 각 그룹을 single stream으로 공급할 수 있다.
대안으로, 모든 파일을 모든 스트림에 공급할 수도 있지만 간단하게 파일의 순서를 섞을 수도 있다. 셔플링은 몇 가지 장점들이 있다. 우선 스트림들에 균형적으로 파티션들이 나눠졌는지 고려하지 않아도 되고, 두번째로 스트림 들에서 파일들 사이의 전환을 랜덤화 할 수 있다. 모델의 state를 재설정 하지 않는다면 모델이 이러한 인위적인 경계를 넘어 가상의 무언가를 배우지 못하게 하는데, 이것은 언어 모델에서는 간단한 사실이다. 
~~~python
import random
class MyIterableDataset(ItreableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
    
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for x in data:
            yield x 
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list)
                     for _ in range(self.batch_size)])
    
    def __iter__(self):
        return self.get_streams()

iterable_dataset = MyIterableDataset(data_list, batch_size=4)
loader = DataLoder(iterable_dataset, batch_size=None)

for batch in islice(loader, 12):
    print(batch)
~~~
## Going Parallel
모델 학습을 할 때, 모델의 forward/backward 패스가 보다, 학습 스피드를 위한 병목은 데이터 로딩일 수 있다. 만약 새로운 데이터를 기다리면, GPU를 사용 했을 때의 많은 이점이 없을 수도 있다. IterableDataset의 대부분의 사용 사례에서 전체 데이터 셋을 메모리에 캐싱하지 않는 것을 전제로 했으니, 이 것을 위한 잠재적인 해결책으로서 병렬로 데이터 로딩을 처리하는 방법을 알아 본다. 
좋은 소식은 Pytorch는 병렬 데이터 로딩이 쉽다는 것이다. 그냥 데이터 로더의 num_workers를 늘려주면 된다. IterableDataset을 병렬로 호출할 때 반환되는 데이터가 예상한 데이터 인지 확인하기 위해 고려 해야 할 몇 가지 주의 사항이 있다.
~~~python
from torch.utils.data import DataLoader, Dataset, IterableDataset
import time 

class MyMapDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self, data):
        return len(self.data)
    
    def __getitem__(self, idx):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1

        start = time.time()
        time.sleep(0.1)
        end = time.time()

        return self.data[idx], worker_id, start, end

data = [i for i in range(17)]
map_dataset = MyMapDataset(data)
loader = DataLoader(map_dataset, batch_size=4, num_workers=0)
plot_timings(loaderr, model_time=0.2, n_batches=4)
~~~
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for x in self.data:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1

            start = time.time()
            time.sleep(0.1)
            end = time.time()

            yield x, worker_id, start, end

iterable_dataset = MyIterableDataset(data)
~~~
~~~python 
import random
from itertools import chain, cycle

class MyIterableDataset(IterableDataset):

    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
    
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for x in data:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1

            start = time.time()
            time.sleep(0.1)
            end = time.time()

            yield x, worker_id, start, end
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list)
                     for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

data_list = [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
    [40, 41, 42, 43],
    [50, 51, 52, 53],
    [60, 61, 62, 63],
    [70, 71, 72, 73],
    [80, 81, 82, 83],
    [90, 91, 92, 93],
]
iterable_dataset = MyIterableDataset(data_list, batch_size=4)
loader = DataLoader(iterable_dataset, batch_size=None, num_workers=2)
~~~
~~~python
class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size

    @property
    def shuffled_data_list(self):
        return sample.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for x in data:
            worker = torch.utils.data.get_worker_info()
            worker_id = id(self) if worker is not None else -1
        
            start = time.time()
            time.sleep(0.1)
            end = time.time()

            yield x, worker_id, start, end 

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffle_data_list)
                    for _ in range(self.batch_size)])
    
    def __iter__(self):
        return self.get_streams()

    @classmethod
    def spilt_datasets(cls, data_list, batch_size, max_workers):
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break 
        split_size = batch_size // num_workers

        return [cls(data_list, batch_size=split_size) 
                for _ in range(num_workers)]

class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def get_stream_loaders(self):
        return zip(*[DataLoaderr(dataset, num_workers=1, batch_size=None)
                    for dataset in datasets])
    
    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))
~~~

import threading
import concurrent.futures
import multiprocessing as mp
from typing import ValuesView

def dummy_free(_):
    pass

def dummy_mem(res):
    return len(res)

# аллокатор подгружает ресурсы только при запросе к элементу с помощью переданной функции
# автоматически освобождает давно неиспользуемые ресурсы при нехватке памяти
class ResourceAllocator:
    def __init__(self, elems_per_block, view_radius, max_memory,
        request_func, free_func = dummy_free, memory_func = dummy_mem):

        self.elems_per_block = elems_per_block
        # максимальное количество кэшированных блоков
        self.max_memory = max_memory
        self.used_memory = 0
        self.used_memory_mutex = threading.Lock()
        # массив блоков[[данные, кэшированное значение], event]
        self.blocks = []
        # индексы неактивных блоков в порядке возрастая их актуальности 
        self.blocks_history = []
        # индексы блоков, находящиеся рядом от последнего запрошенного 
        self.current = 0
        self.count = 0
        self.view_radius = view_radius
        self.request_func = request_func
        self.free_func = free_func
        self.memory_func = memory_func
        num_cores = mp.cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_cores)


    def append(self, resources):
        i = self.count
        for res in resources:
            if i % self.elems_per_block == 0:
                self.blocks.append([[], threading.Event()])
            self.blocks[i // self.elems_per_block][0].append([res, None]) 
            i = i + 1

        self.count = i
    
    def get(self, index):
        if(not (0 <= index < self.count)):
            return None

        # добавляем в историю элементы с прошлой индексации

        for i in range(1, self.view_radius):
            self.__history_block(self.current - i)
        self.__history_block(self.current)
        for i in range(1, self.view_radius):
            self.__history_block(self.current + i)

        self.current = index // self.elems_per_block 

        # делаем блоки активными возле запрошенного индекса, подгружая кэш при необходимости
        self.__renew_block(self.current)
        for i in range(1, self.view_radius):
            self.__renew_block(self.current + i)
        for i in range(1, self.view_radius):
            self.__renew_block(self.current - i)

        # дожидаемся, пока ресурс будет загружен
        self.blocks[self.current][1].wait()

        # стираем старые блоки, если превышен лимит кэширования
        n_removes = 0
        free_mem = 0

        for i in self.blocks_history:
            if(self.used_memory - free_mem > self.max_memory):
                self.blocks[i][1].wait()

                results = [elem[1] for elem in self.blocks[i][0] if elem[0] is not None]
                free_mem += self.memory_func(results)
                self.free_func(results)
                for e in self.blocks[i][0]:
                    e[1] = None
                n_removes += 1
        self.blocks_history = self.blocks_history[n_removes:]
        # if n_removes > 0:
        #     # вывести статистику по активным/неактивным блокам
        #     print("----------------------------------------------------------")
        #     for block in self.blocks:
        #         if(block[0][0][1] is not None):
        #             print("Yes")
        #         else:
        #             print("None")
        
        with self.used_memory_mutex:
            self.used_memory -= free_mem

        return self.blocks[self.current][0][index % self.elems_per_block][1]
    

    def __load_block(self, block):
        resources = [elem[0] for elem in block[0] if elem[0] is not None]
        for elem in block[0]:
            elem[1] = "loading"

        def call_func():
            block[1].clear()
            results = self.request_func(resources)
            memory_usage = self.memory_func(results)
            with self.used_memory_mutex:
                self.used_memory += memory_usage

            for res, elem in zip(results, block[0]):
                elem[1] = res

            block[1].set()

        self.executor.submit(call_func)

    def __renew_block(self, id):
        if(0 <= id < len(self.blocks)):
        # если новый элемент не найден в истории, то значит что он еще не в кэше
            if id in self.blocks_history:
                self.blocks_history.remove(id)
            else:
                self.__load_block(self.blocks[id])

    def __history_block(self, id): 
        # если блок не пустой, то заносим его в историю
        if((0 <= id < len(self.blocks)) and (self.blocks[id][0][0][1] is not None)):
            self.blocks_history.append(id)

# ------ GA Programming -----
# 00000 00000부터 11111 11111까지 가장 큰 이진 정수를 GA로 찾기
# 탐색 중에 해집단의 해들이 일정 비율 동일하게 수렴하면 최적 해로
# 수렴했다고 판단하고 탐색을 종료하도록 설계
# ---------------------------

# ----- 제약사항 ------
# pandas 모듈 사용 금지
# random 모듈만 사용, 필요시 numpy 사용 가능
# [chromosome, fitness]로 구성된 list 타입의 해 사용: ["1010", 10]
# population 형태는 다음과 같이 list 타입으로 규정: [["1010", 10], ["0001", 1], ["0011", 3]]
# --------------------

'''

import random

params = {
    'MUT': 5,  # 변이확률
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터
    'POP_SIZE' : 10,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESS' : 3 # 선택압 조절가능한 파라미터로 설정해둠
    # 원하는 파라미터는 여기에 삽입할 것
    }
# ------------------------------


class GA(): # GA 클래스 함수 정의
    def __init__(self, parameters):
        self.params = {}
        for key, value in parameters.items(): 
            self.params[key] = value # 파라미터 key->value 받아오도록 설정

    def get_fitness(self, chromosome):
        fitness = 0
        bits = [int(c) for c in chromosome] # index마다 값 이진수 쪼각쪼각 저장해두기
        for i in range(self.params["RANGE"]) : # 이진수를 십진수로 변환
            fitness += bits[i]*(2**(self.params["RANGE"]-1-i)) # sum 값이 십진수값 = fitness값
        return fitness

    def print_average_fitness(self, population):
        total = 0
        for ch, fit in population:
            total += fit # population의 평균 fitness를 출력
        population_average_fitness = total / len(population)
        print("population 평균 fitness: {}".format(population_average_fitness))

    def sort_population(self, population):
        population.sort(key = lambda x : x[1], reverse=True) # fitness를 기준으로
                                                            # population을 내림차순 정렬하고 반환
        return population

    def selection_operater(self, population):
        # 토너먼트 선택연산
        def tourna() :
            candidate = random.sample(population, self.params['SELECTION_PRESS'])
            candi_fit = []
            for i in range(self.params['SELECTION_PRESS']) :
                candi_fit.append(candidate[i][1])
            for x,y in candidate :
                if y == max(candi_fit) :
                    return x
        mom_ch = tourna()
        dad_ch = tourna()

        return mom_ch, dad_ch

    def crossover_operater(self, mom_cho, dad_cho):
        # point crossover 구현
        point = random.randint(1, self.params["RANGE"] - 1)  # 1 ~ self.params["RANGE"]-1
        offspring_cho = mom_cho[:point] + dad_cho[point:]
        return offspring_cho

    def mutation_operater(self, chromosome):        
        # 랜덤하게 지정된 하나의 gene를 반대의 값(0->1, 1->0)으로 변이
        idx = random.randint(0, self.params["RANGE"] - 1)
        bits = list(chromosome)
        bits[idx] = '1' if bits[idx] == '0' else '0'
        result_chromosome = ''.join(bits)
        return result_chromosome

    def replacement_operator(self, population, offsprings):
        # 생성된 자식해들(offsprings)을 이용하여 기존 해집단(population)의 해를 대치
        result_population = population + offsprings
        self.sort_population(result_population)
        return result_population[:self.params["POP_SIZE"]]

    # 해 탐색(GA) 함수
    def search(self):
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단        

        # 1. 초기화: 랜덤하게 해를 초기화
        for i in range(self.params["POP_SIZE"]):
            cho = ''.join(random.choice('01') for x in range(self.params["RANGE"]))
            population.append([cho, self.get_fitness(cho)])
            
            self.sort_population(population)         
        print("initialzed population : \n", population, "\n\n")

        while 1:
            offsprings = []
            generation += 1
            for i in range(self.params["NUM_OFFSPRING"]):
                              
                # 2. 선택 연산
                mom_ch, dad_ch = self.selection_operater(population)

                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                if random.randint(1, 100) <= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)

                offsprings.append([offspring, self.get_fitness(offspring)])

            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)

            self.print_average_fitness(population) # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능

            # 6. 알고리즘 종료 조건 판단
            # todo population이 전체 중 self.params["END"]의 비율만큼 동일한 해를 갖는다면
            # 수렴했다고 판단하고 탐색 종료
            cnt=0
            for x in population :
                if x[0] == "1111111111" :
                    cnt+=1
            if cnt / self.params["POP_SIZE"] >= self.params["END"] :
                break


        # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
        print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))


if __name__ == "__main__":
    ga = GA(params)
    ga.search()


'''

# ---------------------------
# GA Programming
# 00000 00000부터 11111 11111까지 가장 큰 이진 정수를 GA로 찾기
# 탐색 중에 해집단의 해들이 일정 비율 동일하게 수렴하면 최적 해로
# 수렴했다고 판단하고 탐색을 종료하도록 설계
# ---------------------------

# ----- 제약사항 ------
# pandas 모듈 사용 금지
# random 모듈만 사용, 필요시 numpy 사용 가능 (여기서는 미사용)
# [chromosome, fitness]로 구성된 list 타입의 해 사용: ["1010", 10]
# population 형태는 list 타입: [["1010", 10], ["0001", 1], ["0011", 3]]
# --------------------

import random

params = {
    'MUT': 5,               # 변이확률(%) 1~100
    'END': 0.9,             # population 내 최빈 해 비율이 이 값 이상이면 종료
    'POP_SIZE': 10,         # population size 10 ~ 100
    'RANGE': 10,            # chromosome bit length (예: 10이면 0000000000 ~ 1111111111)
    'NUM_OFFSPRING': 5,     # 한 세대에 발생하는 자식 chromosome의 수
    'TOUR_T': 0.85,         # 토너먼트 선택의 t
    # 원하는 파라미터는 여기에 삽입할 것
}

class GA:
    def __init__(self, parameters):
        self.params = dict(parameters)

    def get_fitness(self, chromosome):
        # 이진 문자열을 십진수로 변환한 값을 fitness로 사용 (최대화 문제)
        fitness = 0
        bits = [int(c) for c in chromosome]
        for i in range(self.params["RANGE"]):
            fitness += bits[i] * (2 ** (self.params["RANGE"] - 1 - i))
        return fitness

    def print_average_fitness(self, population):
        total = 0
        for _, fit in population:
            total += fit
        population_average_fitness = total / len(population)
        print("population 평균 fitness: {}".format(population_average_fitness))

    def sort_population(self, population):
        population.sort(key=lambda x: x[1], reverse=True)
        return population

    # r < t 이면 좋은 해 선택, 아니면 나쁜 해 선택
    def selection_operater(self, population):
        t = self.params['TOUR_T']

        def tourna():
            x1, x2 = random.sample(population, 2)  # 두 염색체 선택 (중복 없이)
            if x1[1] >= x2[1]:
                good, bad = x1, x2
            else:
                good, bad = x2, x1

            r = random.random()  # [0,1)
            if r < t:
                return good[0]  # chromosome
            else:
                return bad[0]   # chromosome

        mom_ch = tourna()
        dad_ch = tourna()
        return mom_ch, dad_ch

    def crossover_operater(self, mom_cho, dad_cho):
        # point crossover
        point = random.randint(1, self.params["RANGE"] - 1)
        offspring_cho = mom_cho[:point] + dad_cho[point:]
        return offspring_cho

    def mutation_operater(self, chromosome):
        # 랜덤하게 지정된 하나의 gene를 반대로 변이 (0->1, 1->0)
        idx = random.randint(0, self.params["RANGE"] - 1)
        bits = list(chromosome)
        bits[idx] = '1' if bits[idx] == '0' else '0'
        return ''.join(bits)

    def replacement_operator(self, population, offsprings):
        # (μ + λ) 방식: 기존 + 자식 합쳐서 상위 POP_SIZE 유지
        result_population = population + offsprings
        self.sort_population(result_population)
        return result_population[:self.params["POP_SIZE"]]

    # 종료조건: population 내 최빈 chromosome의 비율이 END 이상이면 종료
    def convergence_ratio(self, population):
        freq = {}
        for ch, _ in population:
            freq[ch] = freq.get(ch, 0) + 1

        best_cnt = 0
        best_ch = None
        for ch, cnt in freq.items():
            if cnt > best_cnt:
                best_cnt = cnt
                best_ch = ch

        ratio = best_cnt / self.params["POP_SIZE"]
        return best_ch, best_cnt, ratio

    def search(self):
        generation = 0
        population = []

        # 1. 초기화
        for _ in range(self.params["POP_SIZE"]):
            cho = ''.join(random.choice('01') for _ in range(self.params["RANGE"]))
            population.append([cho, self.get_fitness(cho)])
        self.sort_population(population)

        print("initialzed population : \n", population, "\n\n")

        while True:
            generation += 1
            offsprings = []

            for _ in range(self.params["NUM_OFFSPRING"]):
                # 2. 선택 연산 (교재 토너먼트)
                mom_ch, dad_ch = self.selection_operater(population)

                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                if random.randint(1, 100) <= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)

                offsprings.append([offspring, self.get_fitness(offspring)])

            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)

            # 수렴 관찰용 출력
            self.print_average_fitness(population)

            # 6. 종료 조건
            conv_ch, conv_cnt, conv_ratio = self.convergence_ratio(population)
            if conv_ratio >= self.params["END"]:
                break

        # 최종 출력
        best = population[0]
        conv_ch, conv_cnt, conv_ratio = self.convergence_ratio(population)
        print(
            "탐색이 완료되었습니다.\t 최종 세대수: {},\t 최종 해(상위1): {},\t 최종 적합도: {}\n"
            "수렴 해: {}, 수렴 개수: {}/{}, 수렴 비율: {:.3f}".format(
                generation, best[0], best[1],
                conv_ch, conv_cnt, self.params["POP_SIZE"], conv_ratio
            )
        )

if __name__ == "__main__":
    ga = GA(params)
    ga.search()














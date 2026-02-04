# ------ GA Programming -----
# 00000 00000부터 11111 11111까지 가장 큰 이진 정수를 GA로 찾기
# 탐색 중에 해집단의 해들이 일정 비율 동일하게 수렴하면 최적 해로
# 수렴했다고 판단하고 탐색을 종료하도록 설계
# ---------------------------

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

        first_opt_gen = None  # [추가] 최적해 최초 등장 세대 기록

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
            cnt=0
            for x in population :
                if x[0] == "1111111111" :
                    cnt+=1

            # [추가] 세대별 optimal 비율 로그 + 최초 등장 세대 기록
            print("optimal: {}/{}".format(cnt, self.params["POP_SIZE"]))
            if first_opt_gen is None and cnt > 0:
                first_opt_gen = generation

            if cnt / self.params["POP_SIZE"] >= self.params["END"] :
                break


        # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
        print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))

        # [추가] 최적해 최초 등장 세대 출력
        print("최적해 최초 등장 세대: {}".format(first_opt_gen))


if __name__ == "__main__":
    ga = GA(params)
    ga.search()

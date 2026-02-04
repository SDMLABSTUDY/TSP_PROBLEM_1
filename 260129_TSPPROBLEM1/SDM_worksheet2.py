import random
import math
import time
from collections import Counter

# 파라미터 설정
params = {
    'MUT_RATE': 0.2,
    'POP_SIZE': 80,
    'NUM_OFFSPRING': 40,
    'TOURNAMENT_T': 0.6,
    'CONV_CHECK': 10,  # 몇 세대마다 수렴 검사할지(시간 제한이 있으니까)
    'CONV_MIN_GEN': 50,  # 너무 이른 조기종료 방지용 최소 세대
    'CONV_DOM_RATIO': 0.90,  # 90% 이상이면 수렴으로 판단
}


class TSP_GA:
    def __init__(self, file_path, params, seed=0):
        self.params = params
        self.rng = random.Random(seed)

        self.cities = []
        self.limit_time = 0
        self.n_cities = 0

        self.load_data(file_path)
        self.dist_matrix = self.precompute_distances()

    def load_data(self, file_path):
        """
        입력 형식:
        - 첫 줄: N
        - 다음 N줄: x y
        - 마지막 줄: time_limit (초)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        # 1) 첫 줄: N
        self.n_cities = int(float(lines[0]))

        # 2) 마지막 줄: time limit
        self.limit_time = float(lines[-1])

        # 3) 좌표 N줄 파싱
        coord_lines = lines[1:1 + self.n_cities]
        self.cities = []
        for i, ln in enumerate(coord_lines):
            parts = ln.split()
            if len(parts) != 2:
                raise ValueError(f"{i + 1}번째 좌표 줄이 'x y' 형식이 아닙니다: '{ln}'")
            x, y = map(float, parts)
            self.cities.append((x, y))

    def precompute_distances(self):
        matrix = [[0.0] * self.n_cities for _ in range(self.n_cities)]
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                d = math.sqrt((self.cities[i][0] - self.cities[j][0]) ** 2 +
                              (self.cities[i][1] - self.cities[j][1]) ** 2)
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    def get_fitness(self, chromosome):
        # 경로의 총 길이를 계산하고, 적합도(역수 등)를 반환
        distance = 0
        for i in range(self.n_cities):
            a = chromosome[i]
            b = chromosome[(i + 1) % self.n_cities]  # 마지막 -> 시작 포함
            distance += self.dist_matrix[a][b]
        return distance

    def is_valid(self, chromosome):
        return len(chromosome) == self.n_cities and set(chromosome) == set(range(self.n_cities))

    def greedy_solution(self):
        # Nearest Neighbor 알고리즘 구현
        # 1. 시작 도시 결정
        start = 0
        n = self.n_cities
        dm = self.dist_matrix

        visited = [False] * n
        tour = [start]
        visited[start] = True
        cur = start

        # 2. 방문하지 않은 도시 중 가장 가까운 곳 선택 (반복)
        for k in range(n - 1):
            nxt = None
            best_d = float("inf") # GPT : 아직 최소거리가 없으니 무한대로 두고 첫 후보가 무조건 갱신되게 함
            for j in range(n):
                if not visited[j]:
                    d = dm[cur][j]
                    if d < best_d:
                        best_d = d
                        nxt = j
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt

        best_len = self.get_fitness(tour) # 3. 마지막에 시작 도시로 복귀
        print("Greedy Solution 탐색 완료")
        print(f"  [Greedy start=0] len={best_len:.4f}, tour={tour}")
        return tour, best_len

    def random_tour(self):
        t = list(range(self.n_cities))
        self.rng.shuffle(t)
        return t

    def sort_population(self, population):
        population.sort(key=lambda x: x[1])
        return population

    def selection_operater(self, population):
        # 두 염색체 x1, x2를 무작위로 선택
        # 더 좋은 해(거리 더 짧음)를 x1이라 두고,
        # r ~ U[0,1) 생성
        #  if (t > r) x1 선택 else x2 선택

        t = self.params['TOURNAMENT_T']

        def tourna():
            # 1) 두 개체 무작위 선택 (독립추출)
            ch1, fit1 = population[self.rng.randrange(len(population))]
            ch2, fit2 = population[self.rng.randrange(len(population))]

            # 2) x1이 더 좋다고 가정
            if fit2 < fit1:
                ch1, ch2 = ch2, ch1 # 위치를 바꾸어서 더 좋은 놈을 x1으로 두기
                fit1, fit2 = fit2, fit1

            # 3) 확률적 선택
            r = self.rng.random()  # [0,1)
            if t > r:
                return ch1  # 더 좋은 해 선택
            else:
                return ch2  # 덜 좋은 해 선택

        mom_ch = tourna()
        dad_ch = tourna()
        return mom_ch, dad_ch

    def crossover_operater(self, mom, dad):
        # Order Crossover : 순열을 깨지 않으면서 교차 연산
        n = self.n_cities

        l = self.rng.randrange(n)
        r = self.rng.randrange(n)
        while r == l:
            r = self.rng.randrange(n)
        if l > r:
            l, r = r, l

        child = [-1] * n

        # 1) mom의 구간 복사
        child[l:r + 1] = mom[l:r + 1]
        used = set(child[l:r + 1])

        # 2) dad 순서대로 나머지 채우기
        idx = (r + 1) % n # r+1부터 채우기 시작할 것임
        for city in dad:
            if city in used:
                continue # 중복 스킵
            while child[idx] != -1:
                idx = (idx + 1) % n
            child[idx] = city
            used.add(city)

        return child

    def mutation_operater(self, chromosome):
        n = self.n_cities
        a = self.rng.randrange(n)
        b = self.rng.randrange(n)
        if a > b:
            a, b = b, a
        if a == b:
            return chromosome[:] # 그대로

        ch = chromosome[:]
        ch[a:b+1] = reversed(ch[a:b+1]) # GPT,,
        return ch

    def replacement_operator(self, population, offsprings):
        # population + offsprings 합친 뒤 거리 오름차순 정렬
        # 상위 POP_SIZE 다음 세대로 유지

        merged = population + offsprings
        merged.sort(key=lambda x: x[1])
        return merged[:self.params['POP_SIZE']]

    def print_average_fitness(self, population):
        avg = sum(f for _, f in population) / len(population)
        best = population[0][1]
        print(f"[Gen] avg_len={avg:.4f}, best_len={best:.4f}")

    def search(self):
        generation = 0
        population = []  # [(chromosome, fitness), ...]
        offsprings = []  # [(chromosome, fitness), ...]

        # 1) 초기화: 랜덤하게 해 생성
        for i in range(self.params["POP_SIZE"]):
            ch = self.random_tour()
            fit = self.get_fitness(ch)
            population.append((ch, fit))

        # 정렬(거리 최소화이므로 오름차순)
        population = self.sort_population(population)
        print("initialized population (top 3):")
        for i in range(min(3, len(population))):
            print(f"  {i}: len={population[i][1]:.4f}, tour={population[i][0]}")
        print()

        start_time = time.time()

        # 2) 제한시간 루프 (큰 틀만 제공)
        while (time.time() - start_time) < self.limit_time: # 위험?? 여유시간을 좀 더 두어야...
            generation += 1
            offsprings = []
            # 2-1) 자식 생성
            for _ in range(self.params["NUM_OFFSPRING"]):
                # (a) 선택
                mom_ch, dad_ch = self.selection_operater(population)

                # (b) 교차
                child = self.crossover_operater(mom_ch, dad_ch)

                # (c) 변이 여부 결정
                if self.rng.random() < self.params["MUT_RATE"]:
                    child = self.mutation_operater(child)

                # (d) 유효성 체크(디버깅용)
                # if not self.is_valid(child):
                #     raise ValueError("Invalid tour produced!")

                child_fit = self.get_fitness(child)
                offsprings.append((child, child_fit))

            # 2-2) 대치
            population = self.replacement_operator(population, offsprings)

            # 2-3) (선택) 수렴 관찰용 출력
            if generation % 10 == 0:
                self.print_average_fitness(population)

            # 2-4) 수렴 종료조건
            # population이 일정 비율 동일한 해가 되면 종료

            if generation >= self.params.get("CONV_MIN_GEN", 0) and \
               generation % self.params.get("CONV_CHECK", 10) == 0: # GPT
                # 리스트를 tuple로 변환
                tours = [tuple(ch) for ch, fit in population]
                cnt = Counter(tours)
                rat = cnt.most_common(1)[0][1] / len(population)

                if rat >= self.params.get("CONV_DOM_RATIO", 1.0):
                    # 수렴으로 판단 → 조기 종료
                    print(f"[Converged] dominant_ratio={rat:.2f} at gen={generation}")
                    break

        # 최종 출력
        best_ch, best_fit = population[0]
        elapsed = time.time() - start_time
        print("\n탐색이 완료되었습니다.")
        print(f"최종 세대수: {generation}")
        print(f"소요 시간: {elapsed:.3f}s / 제한 시간: {self.limit_time:.3f}s")
        print(f"최종 해 길이(best_len): {best_fit:.4f}")
        print(f"최종 해(best_tour): {best_ch}")
        return best_ch, best_fit


if __name__ == "__main__":
    tsp = TSP_GA("cycle101.in", params=params)
    tsp.greedy_solution()
    tsp.search()
# @title Best Params without greedy(random initialization)
import random
import math
import time
import statistics
import copy
from collections import Counter
import matplotlib.pyplot as plt
import os
from google.colab import drive


drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/SDMLAB/260129_TSP"

# ==========================================
# 2. TSP_GA 클래스 정의
# ==========================================
class TSP_GA:
    def __init__(self, file_path, params, seed=0):
        self.params = params
        self.rng = random.Random(seed)

        self.cities = []
        self.limit_time = 0
        self.n_cities = 0

        self.load_data(file_path)
        self.dist_matrix = self.precompute_distances()

        self.trace_t = []
        self.trace_best = []

    def load_data(self, file_path):
        """
        입력 형식: N, 좌표들..., TimeLimit
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        self.n_cities = int(float(lines[0]))
        self.limit_time = float(lines[-1])

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
        distance = 0
        for i in range(self.n_cities):
            a = chromosome[i]
            b = chromosome[(i + 1) % self.n_cities]
            distance += self.dist_matrix[a][b]
        return distance

    def greedy_solution(self):
        start = 0
        n = self.n_cities
        dm = self.dist_matrix

        visited = [False] * n
        tour = [start]
        visited[start] = True
        cur = start

        for k in range(n - 1):
            nxt = None
            best_d = float("inf")
            for j in range(n):
                if not visited[j]:
                    d = dm[cur][j]
                    if d < best_d:
                        best_d = d
                        nxt = j
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt

        return tour, self.get_fitness(tour)

    def random_tour(self):
        t = list(range(self.n_cities))
        self.rng.shuffle(t)
        return t

    def sort_population(self, population):
        population.sort(key=lambda x: x[1])
        return population

    def selection_operater(self, population):
        t = self.params['TOURNAMENT_T']

        def tourna():
            ch1, fit1 = population[self.rng.randrange(len(population))]
            ch2, fit2 = population[self.rng.randrange(len(population))]

            if fit2 < fit1:
                ch1, ch2 = ch2, ch1
                fit1, fit2 = fit2, fit1

            r = self.rng.random()
            if t > r:
                return ch1
            else:
                return ch2

        mom_ch = tourna()
        dad_ch = tourna()
        return mom_ch, dad_ch

    def crossover_operater(self, mom, dad):
        n = self.n_cities
        l = self.rng.randrange(n)
        r = self.rng.randrange(n)
        while r == l:
            r = self.rng.randrange(n)
        if l > r:
            l, r = r, l

        child = [-1] * n
        child[l:r + 1] = mom[l:r + 1]
        used = set(child[l:r + 1])

        idx = (r + 1) % n
        for city in dad:
            if city in used:
                continue
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
            return chromosome[:]

        ch = chromosome[:]
        ch[a:b+1] = reversed(ch[a:b+1])
        return ch

    def replacement_operator(self, population, offsprings):
        merged = population + offsprings
        merged.sort(key=lambda x: x[1])
        return merged[:self.params['POP_SIZE']]

    def print_average_fitness(self, population):
        avg = sum(f for _, f in population) / len(population)
        best = population[0][1]
        # 출력 양이 너무 많으면 이 부분 주석 처리 가능
        # print(f"[Gen] avg_len={avg:.4f}, best_len={best:.4f}")

    def search(self):
        generation = 0
        population = []

        # 1) 초기화
        for _ in range(self.params["POP_SIZE"]):
            ch = self.random_tour()
            fit = self.get_fitness(ch)
            population.append((ch, fit))

        population = self.sort_population(population)

        # 2) 시간 및 trace 초기화
        start_time = time.time()
        last_log = 0.0
        self.trace_t = [0.0]
        self.trace_best = [population[0][1]]
        log_interval = self.params.get("LOG_INTERVAL_SEC", 0.2)

        # 3) 제한시간 루프
        while (time.time() - start_time) < self.limit_time:
            generation += 1
            offsprings = []

            for _ in range(self.params["NUM_OFFSPRING"]):
                mom_ch, dad_ch = self.selection_operater(population)
                child = self.crossover_operater(mom_ch, dad_ch)

                if self.rng.random() < self.params["MUT_RATE"]:
                    child = self.mutation_operater(child)

                child_fit = self.get_fitness(child)
                offsprings.append((child, child_fit))

            population = self.replacement_operator(population, offsprings)

            elapsed = time.time() - start_time
            if elapsed - last_log >= log_interval:
                last_log = elapsed
                self.trace_t.append(elapsed)
                self.trace_best.append(population[0][1])

        best_ch, best_fit = population[0]
        elapsed = time.time() - start_time
        return best_ch, best_fit, generation, elapsed, (self.trace_t, self.trace_best)

# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":

    # 3. 대상 파일명 설정
    target_filename = "cycle101.in"

    # 전체 경로 결합
    full_path = os.path.join(base_path, target_filename)

    print(f"Dataset Path: {full_path}")

    # 파일 존재 여부 확인
    if not os.path.exists(full_path):
        print(f"오류: 파일을 찾을 수 없습니다.: {full_path}")
    else:
        print(f"파일을 찾았습니다! 실험을 시작합니다.")


        seeds = [0, 1, 2, 3, 4]

        # 파라미터 설정
        params = {
            'MUT_RATE': 0.25,
            'POP_SIZE': 120,
            'NUM_OFFSPRING': 70,
            'TOURNAMENT_T': 0.8,
            'LOG_INTERVAL_SEC': 0.5, # 로그 간격 조정
        }

        print(f"\n==========================================")
        print(f"[Experiment] {target_filename}")

        # Greedy (Baseline) 계산
        tsp_greedy = TSP_GA(full_path, params, seed=0)
        _, greedy_len = tsp_greedy.greedy_solution()

        # 파일에서 읽어온 제한 시간 확인
        limit_time_val = tsp_greedy.limit_time

        print(f"Greedy(start=0) Len = {greedy_len:.4f}")
        print(f"Time Limit          = {limit_time_val} seconds per seed")
        print("------------------------------------------")
        print(f"{'seed':>4} | {'GA_best_len':>12} | {'improve(%)':>10} | {'Generations':>11}")
        print("-" * 52)

        ga_results = []
        improvements = []

        # 시드별 반복 실행
        for s in seeds:
            # GA 객체 생성
            tsp = TSP_GA(full_path, params, seed=s)

            # 탐색 시작
            best_tour, best_len, gen, elapsed, trace = tsp.search()

            # 개선율 계산
            imp = (greedy_len - best_len) / greedy_len * 100.0

            ga_results.append(best_len)
            improvements.append(imp)

            print(f"{s:>4} | {best_len:>12.4f} | {imp:>10.2f} | {gen:>11}")

        # 모든 시드 종료 후 최종 요약 출력
        mean_len = statistics.mean(ga_results)
        stdev_len = statistics.stdev(ga_results) if len(seeds) > 1 else 0.0
        mean_imp = statistics.mean(improvements)
        stdev_imp = statistics.stdev(improvements) if len(seeds) > 1 else 0.0

        print("-" * 52)
        print(f"[Summary] {target_filename} (5 Runs)")
        print(f"GA_best_len (Mean ± Std) : {mean_len:.4f} ± {stdev_len:.4f}")
        print(f"Improvement (Mean ± Std) : {mean_imp:.2f}% ± {stdev_imp:.2f}%")
        print("==========================================\n")
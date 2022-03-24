# 1018 체스판 다시 칠하기
# from sys import stdin
# n, m = map(int, stdin.readline().split())
# board = []
# for i in range(n):
#     board.append(stdin.readline().strip())
# result = []
# for i in range(n-7):
#     for j in range(m-7):
#         first_W = 0
#         first_B = 0
#         for k in range(i, i+8):
#             for l in range(j, j+8):
#                 if (k + l) % 2 == 0:
#                     if board[k][l] != 'W':
#                         first_W = first_W + 1
#                     if board[k][l] != 'B':
#                         first_B = first_B + 1
#                 else:
#                     if board[k][l] != 'B':
#                         first_W = first_W + 1
#                     if board[k][l] != 'W':
#                         first_B = first_B + 1
#         result.append(first_W)
#         result.append(first_B)
#
# print(min(result))

# 1085 직사각형에서 탈출
# from sys import stdin
# x, y, w, h = map(int, stdin.readline().split())
# print(min(x, y, w-x, h-y))

# 1181 단어 정렬
# from sys import stdin
# N = int(stdin.readline())
# data = list()
# size = list()
# for n in range(N):
#     data.append(stdin.readline().strip())
# data = list(set(data))  # 중복 제거
# data.sort()
# data.sort(key=len)
# for d in data:
#     print(d)

# 1259 팰린드롬수
# import math
# from sys import stdin
# while True:
#     data = list(stdin.readline().strip())
#     length = len(data)
#     count = 0
#     if data == ['0']:
#         break
#     else:
#         for i in range(math.trunc(length / 2)):
#             if data[i] == data[-(i+1)]:
#                 count += 1
#         if count == math.trunc(length / 2):
#             print("yes")
#         else: print("no")

# 1436 영화감독 숌
# from sys import stdin
# n = int(stdin.readline())
# count = 0
# result = 666
# while True:
#     if '666' in str(result):
#         count += 1
#     if count == n:
#         print(result)
#         break
#     result += 1

# 1654 랜선 자르기
# from sys import stdin

# 1929 소수 구하기
# from sys import stdin
# import math
# M, N = map(int, stdin.readline().split())
# if M == 1:
#     M += 1
#
#
# def decimal(n):
#     count = 0
#     for i in range(2, int(math.sqrt(n)) + 1):
#         if n % i == 0:
#             count += 1
#             break
#     if not count: print(n)
#
#
# for i in range(M, N+1):
#     decimal(i)

# 1966 프린터 큐
# from sys import stdin
# K = int(stdin.readline())
# for k in range(K):
#     n, m = map(int, stdin.readline().split())  # n은 문서 개수 m은 몇번째 인쇄됐는지 찾아야하는 문서의 인덱스
#     imp = list(map(int, stdin.readline().strip().split()))
#     idx = list(range(len(imp)))
#     idx[m] = 'target'
#     order = 0
#
#     while True:
#         if imp[0] == max(imp):
#             order += 1
#             if idx[0] == 'target':
#                 print(order)
#                 break
#             else:
#                 imp.pop(0)
#                 idx.pop(0)
#         else:
#             imp.append(imp.pop(0))
#             idx.append(idx.pop(0))

# 1978 소수 찾기
# from sys import stdin
# import math
# def decimal(n):
#     d = 0
#     if n <= 1:
#         return False
#     for i in range(2, int(math.sqrt(n)) + 1):
#         if n % i == 0:
#             d += 1
#             break
#     if d:
#         return False
#     else:
#         return True
#
# N = int(stdin.readline())
# data = list(map(int, stdin.readline().strip().split()))
# n = 0
# for i in data:
#     if decimal(i): n += 1
# print(n)

# 2108 통계학
# from sys import stdin
# from collections import Counter
#
#
# def mode(l):
#     db = Counter(l).most_common(2)  # 카운터의 most common(?) 은 개수가 같은 요소가 여러개일때 오름차순으로 정리하여 보여준다
#     if len(db) > 1:
#         if db[0][1] > db[1][1]:
#             return db[0][0]
#         else:
#             return db[1][0]
#     else:
#         return db[0][0]
#
# 
# N = int(stdin.readline())
# data = list()
# for _ in range(N):
#     data.append(int(stdin.readline()))
# data.sort()
# print(round(sum(data) / N))  # 산술평균
# print(data[N // 2])  # 중앙값: 인덱스는 0부터 시작하므로 2로 나눈 몫이 중앙값의 인덱스가 된다
# print(mode(data))  # 최빈값
# print(max(data) - min(data))


# 2164 카드 2
# from sys import stdin
# from collections import deque
# N = int(stdin.readline())
# q = deque(range(1, N+1))
# for _ in range(N-1):
#     q.popleft()
#     q.append(q.popleft())
#
# print(*q)

# 2231 분해합
# from sys import stdin
# N = int(stdin.readline())
# for i in range(1, N+1):
#     num = sum(map(int, str(i)))  # 문자열은 반복 가능한 객체(iterable) 이므로 map함수의 객체로 들어갈 수 있다 (i의 자릿수 더한값)
#     num_sum = i + num
#     if num_sum == N:
#         print(i)
#         break
#     if i == N:
#         print(0)

# 2292 벌집
# from sys import stdin
# n = int(stdin.readline())
# honeycomb = 1
# cnt = 1
# while n > honeycomb:
#     honeycomb += 6 * cnt # 벌집을 한층 쌓은것
#     cnt += 1
# print(cnt)

# 2609 최소공약수와 최소공배수
# from sys import stdin
# n, m = map(int, stdin.readline().split())
#
#
# def gcd(m, n):
#     if m < n:  # m > n 이 성립하도록
#         m, n = n, m
#     if n == 0:
#         return m
#     if m % n == 0:
#         return n
#     else:
#         return gcd(n, m % n)
#
#
# k = gcd(m, n)
# print(k)
# print(int((m/k)*(n/k)*k))


# 2751 수 정렬하기2
# from sys import stdin
# n = int(stdin.readline())
# l = []
# for _ in range(n):
#     l.append(int(stdin.readline()))
# l.sort()
# for i in range(n):
#     print(l[i])


# 2775 부녀회장이 될테야
# from sys import stdin
# t = int(stdin.readline())
#
# for _ in range(t):
#     f = int(stdin.readline())  # 층수
#     n = int(stdin.readline())  # 호수
#     f0 = [x for x in range(1, n+1)]  # 0층에 사는 사람 정보
#     for k in range(f):
#         for i in range(1, n):
#             f0[i] += f0[i-1]
#
#     print(f0[-1])

# 2798 블랙잭
# from sys import stdin
# from itertools import combinations
# n, m = map(int, stdin.readline().split())
# cards = list(map(int, stdin.readline().strip().split()))  # 각 카드에 적힌 숫자
# com = combinations(cards, 3)
# result = 0
# for i in com:
#     if m+1 > sum(i):
#         result = max(sum(i), result)
#
# print(result)


# 4153 직각삼각형
# from sys import stdin
# while True:
#     t = list(map(int, stdin.readline().strip().split()))
#     if t == [0, 0, 0]:
#         break
#     t.sort()
#     if t[-1] * t[-1] == (t[0] * t[0]) + (t[1] * t[1]):
#         print("right")
#     else: print("wrong")

# 9012 괄호
# from sys import stdin
# t = int(stdin.readline())
# for i in range(t):
#     data = list(stdin.readline().strip())
#     stack = []
#     s = 1
#     if data.count("(") != data.count(")"):
#         print("NO")
#         continue
#     for j in range(len(data)):
#         if data[j] == "(":
#             stack.append("(")
#         if data[j] == ")":
#             if stack:
#                 stack.pop()
#             else:
#                 print("NO")
#                 s -= 1
#                 break
#     if s:
#         if stack:
#             print("NO")
#         else:
#             print("YES")

# 10250 acm 호텔
# from sys import stdin
# t = int(stdin.readline())
# for _ in range(t):
#     h, w, n = map(int, stdin.readline().split())
#     if n % h == 0:
#         k = n // h
#         floor = h
#     else:
#         k = (n // h) + 1
#         floor = n % h
#
#     print(floor * 100 + k)

# 10814 나이순 정렬
# from sys import stdin
# n = int(stdin.readline())
# data = []
# for _ in range(n):
#     age, name = stdin.readline().split()
#     age = int(age)
#     data.append((age, name))
#
# data.sort(key=lambda x: x[0])  # x[0]만 기준삼아 정렬 - stable 정렬 (중복된것 순서 바꾸지 않는 정렬)
# for i in data:
#     print(*i)

# 10816 숫자 카드 2 - 이분탐색이 효과적으로 적용되지 못하는 경우(탐색대상이 중복, 여러개): 딕셔너리로 접근
# from sys import stdin
# n = int(stdin.readline())
# ans = list()
# data = list(map(int, stdin.readline().strip().split()))
# m = int(stdin.readline())
# search = list(map(int, stdin.readline().strip().split()))
# dic = dict()
#
# for i in data:
#     try:  # 딕셔너리에 값을 추가하고 변경하는 법 - dic[i] = 8 하면 딕셔너리에 i:8 쌍이 추가된다.
#         dic[i] += 1
#     except:
#         dic[i] = 1
#
# for i in search:
#     try:
#         print(dic[i], end=" ")
#     except:
#         print(0, end=" ")

# 11050 이항 계수 1
# from sys import stdin
# n, k = map(int, stdin.readline().split())
# if k < 0 or k > n:
#     print(0)
# else:
#     upper = 1
#     lower = 1
#     for i in range(k):
#         upper *= n
#         n -= 1
#         lower *= (i+1)
#     print(int(upper/lower))

# 11650 좌표 정렬하기
# from sys import stdin
# n = int(stdin.readline())
# data = []
# for _ in range(n):
#     data.append(tuple(map(int, stdin.readline().strip().split())))
# data.sort()
#
# for i in data:
#     print(*i)

# 11866 요세푸스 문제
# from sys import stdin
# from collections import deque
# n, k = map(int, stdin.readline().split())
# q = deque([i for i in range(1, n+1)])
# print("<", end="")
# while q:
#     for i in range(k-1):  # k-1번동안 왼쪽을 뽑아 오른쪽에 추가한다
#         q.append(q.popleft())
#     print(q.popleft(), end="")  # k 번째로 뽑은 왼쪽은 print
#     if q:
#         print(", ", end="")
# print(">")

# 1260 dfs 와 bfs
# from sys import stdin
# from collections import deque
# n, m, v = map(int, stdin.readline().split())  # 노드 수, 간선 수, 시작점
# graph = [[] for _ in range(n+1)]
# for i in range(m):  # 각 노드에 연결된 노드를 표현하는 graph 를 채우는 과정
#     start, end = map(int, stdin.readline().split())
#     graph[start].append(end)
#     graph[end].append(start)
# for o in graph:
#     o.sort()
# visited_dfs = [False]*(n+1)  # dfs 에 사용할 방문표시 리스트 - 그래프와 마찬가지로 0번을 비워둠
# visited_bfs = [False]*(n+1)  # bfs 에 사용할 방문표시 리스트
#
#
# def dfs(graph, v, visited):
#     visited[v] = True  # 현재 노드를 방문처리
#     print(v, end=" ")
#     for j in graph[v]:  # 현재 노드와 연결된 모든 노드를 재귀적으로 방문함
#         if not visited[j]:  # j 노드가 false 일 경우 실행하여 방문처리를 하도록 함
#             dfs(graph, j, visited)  # j 를 현재 노드로 하여 재귀적으로 방문실행
#
#
# def bfs(graph, v, visited):
#     queue = deque([v])  # 덱을 만들때부터 시작노드를 넣어놓고 시작하는것(v)
#     visited[v] = True  # 현재 노드 방문처리
#     while queue:  # 큐가 빌때까지 반복
#         now = queue.popleft()
#         print(now, end=" ")
#         for k in graph[now]:
#             if not visited[k]:
#                 queue.append(k)
#                 visited[k] = True
#
#
# dfs(graph, v, visited_dfs)
# print()
# bfs(graph, v, visited_bfs)

# 1697 숨바꼭질
# from sys import stdin
# from collections import deque
# n, k = map(int, stdin.readline().split())  # 수빈위치, 동생위치
# MAX = 10 ** 5
# distance = [0] * (MAX + 1)
#
#
# def bfs():
#     queue = deque([n])
#     while queue:
#         x = queue.popleft()
#         if x == k:
#             print(distance[x])
#             break
#         for nx in (x-1, x+1, x*2):
#             if 0 <= nx <= MAX and not distance[nx]:  # 다음 방문할 노드가 n,k의 조건을 넘지 않고 방문 하지 않은 상태일때 실행
#                 distance[nx] = distance[x] + 1  # v까지 거리에 1을 더함
#                 queue.append(nx)
#
#
# bfs()

# 1012 유기농 배추 (dfs)
# import sys
# from sys import stdin
# sys.setrecursionlimit(10000)
# t = int(stdin.readline())
# dx = [1, -1, 0, 0]  # 상하좌우의 인덱스를 표현
# dy = [0, 0, -1, 1]
#
#
# def dfs(a, b):  # 해당노드와 인접한 노드를 재귀적으로 모두 방문처리
#     s[a][b] = 0  # 해당노드 방문처리
#     for i in range(4):  # 상하좌우 체크
#         nx = a + dx[i]
#         ny = b + dy[i]
#         if 0 <= nx < n and 0 <= ny < m and s[nx][ny] == 1:  # 해당 좌표가 배열 내에 있는지 & 해당 좌표에 배추가 있는지
#             dfs(nx, ny)
#
#
# for i in range(t):
#     m, n, k = map(int, stdin.readline().split())  # 가로 세로 배추개수
#     s = [[0] * m for i in range(n)]  # 그래프 생성
#     cnt = 0
#     for j in range(k):  # 배추 심기
#         y, x = map(int, stdin.readline().split())
#         s[x][y] = 1
#     for o in range(n):
#         for t in range(m):
#             if s[o][t] == 1:
#                 dfs(o, t)
#                 cnt += 1
#     print(cnt)

# 1012 유기농 배추 (bfs)
# from sys import stdin
# from collections import deque
# t = int(stdin.readline())
# dx = [1, -1, 0, 0]  # 상하좌우의 인덱스를 표현
# dy = [0, 0, -1, 1]
#
#
# def bfs(a, b):
#     q = deque()
#     q.append([a, b])
#     s[a][b] = 0  # 현재 노드 방문처리
#     while q:
#         v = q.popleft()
#         for num in range(4):
#             nx = v[0] + dx[num]
#             ny = v[1] + dy[num]
#             if 0 <= nx < n and 0 <= ny < m:
#                 if s[nx][ny] == 1:
#                     s[nx][ny] = 0
#                     q.append([nx, ny])
#
#
# for i in range(t):
#     m, n, k = map(int, stdin.readline().split())  # 가로 세로 배추개수
#     s = [[0] * m for i in range(n)]  # 그래프 생성
#     cnt = 0
#     for j in range(k):  # 배추 심기
#         y, x = map(int, stdin.readline().split())
#         s[x][y] = 1
#     for o in range(n):
#         for t in range(m):
#             if s[o][t] == 1:
#                 bfs(o, t)
#                 cnt += 1
#     print(cnt)

# 7576 토마토
# from sys import stdin
# from collections import deque
# m, n = map(int, stdin.readline().split())
# graph = []
# q = deque()
# for _ in range(n):  # 그래프 채워넣기
#     graph.append(list(map(int, stdin.readline().strip().split())))
# for o in range(n):  # 익은 토마토를 탐색하고 그 좌표를 큐에 넣음 - 익은 토마토(1)을 방문처리 된걸로 봄
#     for t in range(m):
#         if graph[o][t] == 1:
#             q.append([o, t])
# dx = [1, -1, 0, 0]  # 상하좌우의 인덱스를 표현
# dy = [0, 0, -1, 1]
# ans = 0
#
#
# # 전체적인 컨셉: 하루에 여러개의 토마토가 동시에 bfs 진행하여 익으므로 오늘 진행해야할 토마토 개수를 cnt_today 에 임시저장
# #              bfs 진행할때마다(하나의 토마토가 주변 토마토 익힘: 하루 걸림) cnt_today 를 -1
# #              cnt_today 가 0이 되는 순간이 하루에 동시 진행해야할 bfs 가 끝나고 하루가 마무리된 것 -> ans +1
# def bfs():
#     global ans
#     cnt_today = len(q)  # 현재 익은 모든 토마토에 대해서 하루에 bfs 를 동시진행 위해 토마토 개수 세는 카운터
#     cnt_tomorrow = 0  # 내일 동시진행 개수 임시저장
#     while q:
#         v = q.popleft()
#         if not cnt_today:  # cnt_today 가 0일경우 하루를 넘기는 과정(ans +1)
#             ans += 1
#             cnt_today = cnt_tomorrow
#             cnt_tomorrow = 0
#         cnt_today -= 1  # 하나의 토마토를 pop 해서 bfs 에 이용했으므로 오늘 bfs 돌릴 토마토 개수 -1
#         for num in range(4):  # 상하좌우 검사
#             nx = v[0] + dx[num]
#             ny = v[1] + dy[num]
#             if 0 <= nx < n and 0 <= ny < m:  # 배열 바깥으로 인덱스가 나가지는 않는지
#                 if not graph[nx][ny]:  # 해당 토마토가 익지 않았다면 실행하여 방문처리
#                     q.append([nx, ny])
#                     graph[nx][ny] = 1  # 방문처리
#                     cnt_tomorrow += 1  # 해당 토마토를 내일이 되면 주변 토마토를 익히는 토마토에 추가
#
#
# bfs()
#
# for o in range(n):  # 아직 익지 않은 토마토가 있는지 검사
#     for t in range(m):
#         if graph[o][t] == 0:
#             ans = -1  # 다 못익히는 경우 답을 -1로 변경
#             break
#
# print(ans)

# 2630 색종이 만들기
# from sys import stdin
# n = int(stdin.readline())
# graph = [list(map(int, stdin.readline().strip().split())) for _ in range(n)]
# cnt_0 = 0
# cnt_1 = 0
#
#
# # 그래프를 직접 쪼개는게 아니고, 하나의 그래프에서 인덱스를 조절해 분할한다.
# def cut(x, y, n):
#     global cnt_0, cnt_1
#     check = graph[x][y]
#     for i in range(x, x+n):
#         for j in range(y, y+n):
#             if check != graph[i][j]:  # 하나라도 다른게 있는지 검사
#                 cut(x, y, n//2)  # //를 쓰는 이유는 마지막에 n이 0이 되도록하여 반복하지 않도록 하기 위함
#                 cut(x, y+n//2, n//2)
#                 cut(x+n//2, y, n//2)
#                 cut(x+n//2, y+n//2, n//2)
#                 return  # 리턴이 있는 이유: 여기까지 들어온것은 하나라도 다른게 있다는 뜻 -> 아래 있는 카운트 증가문 실행하지 않기 위해 리턴 none 으로 함수를 종료시켜버림
#     if check == 0:
#         cnt_0 += 1
#     else:
#         cnt_1 += 1
#
#
# cut(0, 0, n)
# print(cnt_0)
# print(cnt_1)

# 1074 Z
# from sys import stdin
# N, r, c = map(int, stdin.readline().split())
# ans = 0
#
# while N:
#     N -= 1
#     n = 2 ** N
#     if r < n and c < n:
#         pass
#     elif r < n <= c:
#         ans += (n ** 2) * 1
#         c -= n  # 이 사분면을 1사분면으로 만들기
#     elif c < n <= r:
#         ans += (n ** 2) * 2
#         r -= n
#     else:
#         ans += (n ** 2) * 3
#         r -= n
#         c -= n
#
# print(ans)

# 1463 1로 만들기 (bfs) - 시간초과
# from sys import stdin
# from collections import deque
# n = int(stdin.readline())
#
#
# def bfs(x):
#     cnt = 0
#     q = deque([(x, cnt)])
#     while q:
#         x, cnt = q.popleft()
#         if x == 1:
#             return cnt
#         if x % 3 == 0:
#             q.append([x//3, cnt+1])
#         if x % 2 == 0:
#             q.append([x//2, cnt+1])
#         q.append([x-1, cnt+1])
#     return -1
#
#
# print(bfs(n))

# 1463 1로 만들기(dp)
# from sys import stdin
# n = int(stdin.readline())
# dp = [0 for i in range(n+1)]  # dp[i]에는 i까지 오려면 수행해야하는 계산 횟수가 담겨있다
# for i in range(2, n+1):
#     dp[i] = dp[i-1] + 1  # dp[i]가 전 숫자에 1을 더해 만들어진것으로 일단 간주 (계산횟수 +1)
#     if i % 2 == 0 and dp[i] > dp[i//2] + 1:
#         dp[i] = dp[i//2] + 1
#     if i % 3 == 0 and dp[i] > dp[i//3] + 1:
#         dp[i] = dp[i//3] + 1
#
# print(dp[n])

# 1764 듣보잡
# from sys import stdin
# n, m = map(int, stdin.readline().split())
# eme = {stdin.readline().strip() for i in range(n)}
# qh = {stdin.readline().strip() for i in range(m)}
# ans = eme.intersection(qh)
# ans = list(ans)
# ans.sort()
# print(len(ans))
# for i in ans:
#     print(i)

# 1927 최소 힙
# from sys import stdin
# import heapq
# n = int(stdin.readline())
# heap = []
# for i in range(n):
#     k = int(stdin.readline())
#     if k > 0:
#         heapq.heappush(heap, k)
#     else:
#         try:
#             print(heapq.heappop(heap))
#         except IndexError:
#             print(0)

# 1931 회의실 배정
# from sys import stdin
# n = int(stdin.readline())
# time = [[0]*2 for _ in range(n)]
# for i in range(n):
#     s, e = map(int, stdin.readline().split())
#     time[i][0] = s
#     time[i][1] = e
#
# time.sort(key=lambda x: (x[1], x[0]))
# cnt = 0
# end_t = 0
# for i, j in time:
#     if i >= end_t:
#         cnt += 1
#         end_t = j
#
# print(cnt)

# 2606 바이러스 (dfs)
# from sys import stdin
# n = int(stdin.readline())  # 컴퓨터 대수
# k = int(stdin.readline())  # 간선의 개수
# graph = [[] for _ in range(n+1)]  # graph[0]은 비워두는 자리라서 n+1
# visited = [0 for i in range(n+1)]
# for i in range(k):  # 그래프 채워넣기
#     s, e = map(int, stdin.readline().split())
#     graph[s].append(e)
#     graph[e].append(s)
#
#
# def dfs(x):
#     visited[x] = 1  # 방문처리
#     for i in graph[x]:
#         if not visited[i]:
#             dfs(i)
#
#
# dfs(1)
# print(visited.count(1)-1)

# 2606 바이러스 (bfs)
# from sys import stdin
# from collections import deque
# n = int(stdin.readline())  # 컴퓨터 대수
# k = int(stdin.readline())  # 간선의 개수
# graph = [[] for _ in range(n+1)]  # graph[0]은 비워두는 자리라서 n+1
# visited = [0 for i in range(n+1)]
# for i in range(k):  # 그래프 채워넣기
#     s, e = map(int, stdin.readline().split())
#     graph[s].append(e)
#     graph[e].append(s)
#
#
# def bfs(x):
#     q = deque([x])
#     visited[x] = 1
#     while q:
#         v = q.popleft()
#         for i in graph[v]:
#             if not visited[i]:
#                 visited[i] = 1
#                 q.append(i)
#
#
# bfs(1)
# print(visited.count(1)-1)

# 7662 이중 우선순위 큐  *** 꼭 다시 풀어볼 것
# https://neomindstd.github.io/%EB%AC%B8%EC%A0%9C%ED%92%80%EC%9D%B4/boj7662/
# import sys;read = sys.stdin.readline
# import heapq
# result = []  # 테스트케이스 수
# for T in range(int(read())):
#     visited = [False] * 1_000_001  # n이 최대힙,최소힙에 존재 하는지?
#     minH, maxH = [], []  # maxH와 minH의 [0]은 각각 최대값과 최소값이다
#     for i in range(int(read())):
#         s = read.split()  # 자동으로 리스트에 저장된다
#         if s[0] == "I":  # 인서트 연산
#             heapq.heappush(minH, (int(s[1]), i))  # minH 에 (int(s[1]), i) 튜플형태 삽입 (i 는 식별자)
#             heapq.heappush(maxH, (-int(s[1]), i))  # 최소힙을 최대힙으로 바꾸려고 -1을 곱함
#             visited[i] = True  # 최소(최대)힙에 i가 있음을 나타냄
#         elif s[1] == "1":  # 딜리트 연산의 명령이 1인 경우(최대값 삭제)
#             while maxH and not visited[maxH[0][1]]:  # max[0][1]은 최대힙에서 최대값의 식별자을 의미
#                 heapq.heappop(maxH)
#             if maxH:
#                 visited[maxH[0][1]] = False  # 방문처리를 취소함으로써 더이상 리스트에 없음을 나타냄
#                 heapq.heappop(maxH)  # 최대값 삭제처리
#         else:  # 딜리트 연산의 명령이 0인경우(최소값 삭제)
#             while minH and not visited[minH[0][1]]:  # 삭제 대상 노드가 나올때까지 힙에서 제거
#                 heapq.heappop(minH)
#             if minH:
#                 visited[minH[0][1]] = False
#                 heapq.heappop(minH)
#
#     while minH and not visited[minH[0][1]]: heapq.heappop(minH)  # 최소힙 동기화
#     while maxH and not visited[maxH[0][1]]: heapq.heappop(maxH)  # 최대힙 동기화
#     result.append(f'{-maxH[0][0]} {minH[0][0]}' if maxH and minH else 'EMPTY')
# print('\n'.join(result))

# 최대힙과 최소힙의 동기화는 이전 삭제와 동시에 이루어지지 않고 다음 삭제를 할때 확인 후 동기화가 이루어진다
# 772번째줄 설명: 최대힙 리스트에 당연히 있어야할 최대값이 방문처리가 안되어있다는 것은 최소힙 리스트에서 삭제가 이루어지고
# 방문처리가 취소되었음을 의미 -> 방문처리가 되어있는 노드를 만날때까지 계속 최대힙의 최대값을 pop 하면
# 결국 최소힙에도 동시에 "존재하는" 노드 중 가장 큰 값이 튀어나올 것 -> 그것이 우리가 삭제연산 해야할 최대힙의 최대값
# 777번째줄은 이것의 반대로 이해하면 됨 / 아랫줄은 동기화를 한번씩 더 진행하여 쓰레기 노드를 버림
# (동기화가 다음 연산 시 되기 때문에 마지막 딜리트 연산의 동기화가 되어있지 않은 상태

# 9095 1,2,3 더하기  # dp 문제는 점화식을 찾는게 문제풀이의 90% - 규칙이 안보이면 손으로 써서 찾아보자
# from sys import stdin
# t = int(stdin.readline())
# result = []
# for _ in range(t):
#     n = int(stdin.readline())
#     dp = [1, 2, 4]
#     for i in range(3, n+1):
#         dp.append(dp[i-1] + dp[i-2] + dp[i-3])
#     print(dp[n-1])

# 11279 최대 힙
# from sys import stdin
# import heapq
# n = int(stdin.readline())
# maxH = []
# for i in range(n):
#     k = int(stdin.readline())
#     if k > 0:
#         heapq.heappush(maxH, -k)
#     else:
#         if maxH:
#             print(-heapq.heappop(maxH))
#         else:
#             print(0)

# 11399 ATM  # 내 풀이
# from sys import stdin
# n = int(stdin.readline())
# data = list()
# ans = 0
# for i, d in enumerate(map(int, stdin.readline().strip().split())):  # (i번째 사람, 걸리는 시간)튜플 형태로 삽입
#     data.append((i+1, d))
# data.sort(key=lambda x: x[1])
# for i in range(n):
#     nx = 0
#     for j in range(i):
#         nx += data[j][1]
#     ans += data[i][1] + nx
# print(ans)

# 11399 ATM 간단한 버전
# from sys import stdin
# n = int(stdin.readline())
# data = list(map(int, stdin.readline().strip().split()))
# ans = 0
# data.sort()
# for i in range(n):
#     for j in range(i+1):
#         ans += data[j]
# print(ans)

# 11723 집합
# from sys import stdin
# s = set()
# m = int(stdin.readline())
# for _ in range(m):
#     t = stdin.readline().strip().split()
#     if len(t) == 1:  # 명령어만 있는 경우
#         if t[0] == 'all':
#             s = set(i for i in range(1,21))
#         else:
#             s = set()
#
#     else:
#         cmd, tgt = t[0], int(t[1])
#
#         if cmd == 'add':
#             s.add(tgt)
#         elif cmd == 'check':
#             print(1 if tgt in s else 0)
#         elif cmd == 'remove':
#             s.discard(tgt)
#         elif cmd == 'toggle':
#             if tgt in s:
#                 s.discard(tgt)
#             else:
#                 s.add(tgt)

# 11052 카드 구매하기
# from sys import stdin
# n = int(stdin.readline())  # 구매하려는 카드의 개수
# p = [0] + list(map(int, stdin.readline().strip().split()))
# dp = [0 for _ in range(n+1)]
#
# for i in range(1, n+1):
#     for k in range(1, i+1):
#         dp[i] = max(dp[i], dp[i-k] + p[k])
# print(dp[n])

# 1107 리모컨
# from sys import stdin
# tgt = int(stdin.readline())  # 이동하려는 채널
# m = int(stdin.readline())  # 고장난 버튼 개수
# ans = abs(100-tgt)  # +-버튼 일일이 눌러서 이동하는 경우의 버튼조작수
# if m:
#     broken = set(stdin.readline().strip().split())
# else:
#     broke = set()  # 공집합
#
# for num in range(1000001):  # 작은수에서 큰수로 이동과 그 반대의 경우까지 전수탐색(50만 * 2)
#     for n in str(num):
#         if n in broken:  # 해당 번호가 고장난경우
#             break
#     else:  # 고장난 번호가 없는경우
#         ans = min(ans, len(str(num)) + abs(num - tgt))  # num 에 대해서 최소값을 구함
# print(ans)

# 1389 케빈 베이컨의 6단계 법칙
# from sys import stdin
# from collections import deque
#
#
# def bfs(graph, start):
#     num = [0] * (n+1)
#     visited = [start]
#     queue = deque()
#     queue.append(start)
#
#     while queue:
#         a = queue.popleft()  # a는 시작노드
#         for i in graph[a]:
#             if i not in visited:
#                 num[i] = num[a] + 1  # 'a를 만나기까지 횟수' + 1 거쳐야 i를 만날수 있다
#                 visited.append(i)
#                 queue.append(i)
#     return sum(num)
#
#
# n, m = map(int, stdin.readline().split())  # 사람수, 관계수
# graph = [[] for _ in range(n+1)]  # 관계성 표시한 그래프
# for _ in range(m):
#     a, b = map(int, stdin.readline().split())
#     graph[a].append(b)
#     graph[b].append(a)
#
# result = list()
# for i in range(1, n+1):  # 1번사람부터 n 번 사람까지 bfs 함수 돌리기
#     result.append(bfs(graph,i))
#
# print(result.index(min(result))+1)

# 1463 1로 만들기(dp)
# from sys import stdin
# n = int(stdin.readline())
# make_1 = [0, 0, 1, 1]
# for i in range(4, n+1):
#     make_1.append(make_1[i - 1] + 1)
#     if i % 3 == 0:
#         make_1[i] = min(make_1[i], make_1[i//3] + 1)
#     if i % 2 == 0:
#         make_1[i] = min(make_1[i], make_1[i//2] + 1)
#
# print(make_1[n])

# 1541 잃어버린 괄호
# a = input().split('-')
# nums = []
# for i in a:
#     s = 0
#     p = i.split('+')
#     for j in p:
#         s += int(j)
#     nums.append(s)
# result = nums[0]
# for k in range(1, len(nums)):
#     result -= nums[k]
# print(result)


# 1620 포켓몬
# from sys import stdin
# n, m = map(int, stdin.readline().split())  # 포켓몬수, 질문수
# book = dict()
# for i in range(1, n+1):
#     book[i] = stdin.readline().strip()
#
# reverse_b = dict(map(reversed, book.items()))  # 기존 book과 비교해서 key와 value가 정반대  *********핵심
# for _ in range(m):
#     q = stdin.readline().strip()
#     try:
#         q = int(q)
#         print(book[q])
#     except ValueError:  # q가 문자열인 경우
#         print(reverse_b[q])

# 1676 팩토리얼 0의 개수
# from sys import stdin
#
#
# def f(n):
#     if n == 0:
#         return 1
#     else:
#         return n * f(n-1)
#
#
# n = int(stdin.readline())
# fac = f(n)
# n_list = list(str(fac))
# n_list.reverse()
# cnt = 0
# for i in n_list:
#     if i == '0':
#         cnt += 1
#     else:
#         break
# print(cnt)

# 1780 종이의 개수
# from sys import stdin
# n = int(stdin.readline())
# graph = [list(map(int, stdin.readline().strip().split())) for _ in range(n)]
# num_1, num_0, num_m = 0, 0, 0
#
#
# def div(x, y, n):  # 시작 좌표 xy 와 해당 그래프 한변의 길이 n
#     global num_0, num_m, num_1
#     check = graph[x][y]  # 해당 색종이가 모두 같은 색인지 체크하기 위한 기준
#     for i in range(x, x+n):  # 모든 부분이 같은색인지 체크
#         for j in range(y, y+n):
#             if graph[i][j] != check:
#                 for k in range(3):  # 3x3으로 색종이를 나눈다
#                     for l in range(3):
#                         div(x + k * n//3, y + l * n//3, n//3)
#                 return
#     if check == -1:
#         num_m += 1
#     elif check == 0:
#         num_0 += 1
#     else:
#         num_1 += 1
#
#
# div(0, 0, n)
# print(f'{num_m}\n{num_0}\n{num_1}')

#
















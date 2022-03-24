# 입출력과 사칙연산
# 1
# print("Hello World!")

# 2
# print("강한친구 대한육군")
# print("강한친구 대한육군")

# 3
# print("\    /\\")
# print(" )  ( ')")
# print("(  /  )")
# print(" \(__)|")

# 4
# print("|\\_/|")
# print("|q p|   /}")
# print("( 0 )\"\"\"\\")
# print("|\"^\"`    |")
# print("||_/=\\\\__|")

# 5 - 8
# A, B = map(int, input().split())
# print(A+B)

# 9
# A, B = map(int, input().split())
# print(A+B)
# print(A-B)
# print(A*B)
# print(A//B)
# print(A%B)

# 10
# A, B, C = map(int, input().split())
# print((A+B)%C)
# print(((A%C) + (B%C))%C)
# print((A*B)%C)
# print(((A%C) * (B%C))%C)

# 11 너무 알고리즘화를 많이 시킨듯...?
# A = int(input())
# B = input()
# C = (",".join(B).split(","))
# C.reverse()
# for c in C:
#     print(A*int(c))
# print(A*int(B))

# if 문 연습
# A, B = map(int, input().split())
# if B < 45:
#     if A == 0:
#         A = 23
#     else:
#         A -= 1
#     B += 60
#
# print(A, B - 45)

# while 문 연습
# while True:
#     try:
#         A, B = map(int,input().split())
#         print(A + B)
#     except EOFError:
#         break

# 스택 구현 연습
# stack = []
# result = []
# num = int(input())
# for i in range(num):
#     order = list(map(str, input().split()))
#     if order[0] == "push":
#         stack.append(str(order[1]))
#     elif order[0] == "top":
#         if len(stack) == 0:
#             result.append("-1")
#         else:
#             result.append(str(stack[-1]))
#     elif order[0] == "pop":
#         if len(stack) == 0:
#             result.append("-1")
#         else:
#             result.append(str(stack[-1]))
#             del stack[-1]
#     elif order[0] == "size":
#         result.append(str(len(stack)))
#     elif order[0] == "empty":
#         if len(stack) == 0:
#             result.append("1")
#         else:
#             result.append("0")
# print("\n".join(result))

# 단어 뒤집기
# T = int(input())
# result = []
# for i in range(T):
#     sentence = input().split(" ")
#     tempa = []
#     for n, word in enumerate(sentence):
#         temp = ",".join(word).split(",")
#         temp.reverse()
#         tempa.append("".join(temp))
#         tempa.append(" ")
#     result.append("".join(tempa))
# print("\n".join(result))
# 글자뒤집기 답안
# import sys
#
# for i in range(int(sys.stdin.readline())):
#     # [::-1] 이걸로 입력한 단어로 reverse
#     # .split()으로 공백으로 구분하여 list로 쪼갬
#     word = sys.stdin.readline()[
#            ::-1].split()  # ['yadot', 'yppah', 'ma', 'I'], ['ezirp', 'tsrif', 'eht', 'niw', 'ot', 'tnaw', 'eW']
#     word.reverse()  # ['I', 'ma', 'yppah', 'yadot'] , ['eW', 'tnaw', 'ot', 'niw', 'eht', 'tsrif', 'ezirp']
#
#     # " ".join(list) 로 단어들을 이어 붙임
#     print(' '.join(word))  # I ma yppah yadot , eW tnaw ot niw eht tsrif ezirp

# vps판독기
# import sys
# T = int(input())
# for t in range(T):
#     data = list(sys.stdin.readline()[::])
#     data.remove("\n")
#     if data.count("(") != data.count(")"): print("NO")
#     else:
#         for i in range(int(len(data))):
#             try:
#                 left = data.index("(")
#                 right = data[left:].index(")")
#                 del data[left]
#                 del data[right-1]
#             except ValueError:
#                 pass
#         if len(data) == 0: print("YES")
#         else:
#             print("NO")
# vps판독기 답안
# from sys import stdin
#
# n = int(input())
# for _ in range(n):
#     str_ = stdin.readline().strip()
#     stack = 0
#     for chr_ in str_:
#         if chr_ == '(':
#             stack += 1
#         else:
#             stack -= 1
#             if stack < 0:
#                 break
#     if stack == 0:
#         print('YES')
#     else:
#         print('NO')

# 스택 수열 1874
# stack = []
# result = []
# s = 0  # NO인 경우에 결과 리스트 출력 막는 스위치
# cur = 1  # 현재 스택에 들어갈 숫자
# n = int(input())
# for i in range(n):
#     num = int(input())  # 사용자가 입력한 수
#     while cur <= num:  # 입력된 수 만날때까지 push
#         stack.append(cur)
#         result.append("+")
#         cur += 1
#     if stack[-1] == num:
#         stack.pop()
#         result.append("-")
#     else:
#         print("NO")
#         s += 1
#         break
#
# if s == 0:
#     print("\n".join(result))

# 에디터 1406
# from sys import stdin
# s1 = list(stdin.readline().strip())  # 데이터 입력 후 \n지운뒤 문자단위로 리스트화
# s2 = []
# n = int(stdin.readline()) # 명령개수 입력
#
# for i in range(n):
#     order = stdin.readline()
#     if order[0] == "D":
#         if s2: s1.append(s2.pop())
#     elif order[0] == "L":
#         if s1: s2.append(s1.pop())
#     elif order[0] == "B":
#         if s1: s1.pop()
#     else: s1.append(order[2])
#
# s2.reverse()
# print("".join(s1+s2))

# 큐 10845
# from sys import stdin
# N = int(stdin.readline())
# queue = []
# for n in range(N):
#     order = list(stdin.readline().strip().split())
#     if order[0] == "push":
#         queue.append(order[1])
#     elif order[0] == "pop":
#         if queue: print(queue.pop(0))
#         else: print(-1)
#     elif order[0] == "size":
#         print(len(queue))
#     elif order[0] == "empty":
#         if queue: print(0)
#         else: print(1)
#     elif order[0] == "front":
#         if queue: print(queue[0])
#         else: print(-1)
#     elif order[0] == "back":
#         if queue: print(queue[-1])
#         else: print(-1)

# 요세푸스 문제  1158
# from sys import stdin
# n, k = map(int, stdin.readline().split())
# queue = [i+1 for i in range(n)]
# result = []
# index = 0
# while queue:
#     index += k - 1
#     if index >= len(queue):
#         index %= len(queue)
#     result.append(queue.pop(index))
#
# print("<"+", ".join(map(str, result))+">")

# 덱 10866
# from sys import stdin
# N = stdin.readline()
# deck = []
# for n in range(int(N)):
#     order = list(stdin.readline().split())
#     if order[0] == "push_front":
#         deck.insert(0, order[1])
#     elif order[0] == "push_back":
#         deck.append(order[1])
#     elif order[0] == "pop_front":
#         if deck: print(deck.pop(0))
#         else: print(-1)
#     elif order[0] == "pop_back":
#         if deck: print(deck.pop())
#         else: print(-1)
#     elif order[0] == "size":
#         print(len(deck))
#     elif order[0] == "empty":
#         if deck: print(0)
#         else: print(1)
#     elif order[0] == "front":
#         if deck: print(deck[0])
#         else: print(-1)
#     elif order[0] == "back":
#         if deck: print(deck[-1])
#         else: print(-1)

# 단어뒤집기 2 17413
# from sys import stdin
# S = list(stdin.readline().strip())
# i = 0
# while i < len(S):
#     if S[i] == "<":
#         i += 1  # 다음글자 넘어감
#         while S[i] != ">":
#             i += 1
#         i += 1  # 닫힌괄호 만난후 다음 글자로 넘어감
#     elif S[i].isalnum():  # 알파벳이나 숫자를 만났을 경우
#         start = i  # 시작 인덱스
#         while i < len(S) and S[i].isalnum():
#             i += 1
#         temp = S[start:i]  # temp 에 구한 범위 넣음
#         temp.reverse()
#         S[start:i] = temp
#     else:  # 알파벳도 숫자도 괄호도 아니라면 공백
#         i += 1
#
# print("".join(S))

# 쇠막대기 10799
# from sys import stdin
# data = list(stdin.readline().strip())
# stack = []
# answer = 0
# for d in range(len(data)):
#     if data[d] == "(":
#         stack.append("(")
#     else:
#         if data[d-1] == "(":  #()인경우 (를 없애고 (개수만큼 더해줌
#             stack.pop()
#             answer += len(stack)
#         else:
#             stack.pop()
#             answer += 1
#
# print(answer)

# 오큰수 17298 - 시간초과 실패
# from sys import stdin
# N = int(stdin.readline())
# data = list(map(int, stdin.readline().split()))
# NGE = []
# for n in range(N):
#     temp = []
#     for i in range(n+1, N):
#         if data[n] < data[i]:
#             temp.append(data[i])
#     if temp: NGE.append(temp[0])
#     else: NGE.append(-1)
#
# print(NGE)

# 오큰수 17298 - 인덱스를 담는 스택을 활용해 복잡도를 N^2 에서 Nㅇ 가깝게 만들었다
# from sys import stdin
# N = int(stdin.readline())
# data = list(map(int, stdin.readline().split()))
# stack = []  # 데이터의 인덱스를 저장하는 스택
# NGE = [-1] * N  # 새로 채워넣지 않은 경우 = 오큰수가 없는경우 = -1
# stack.append(0)
# for n in range(1, N):
#     while stack and data[stack[-1]] < data[n]:
#         NGE[stack.pop()] = data[n]
#     stack.append(n)
# print(*NGE)  # 리스트의 압축을 풀어 대괄호를 없애는 *연산자


# 17299 오등큰수
# from sys import stdin
# from collections import Counter
# N = int(stdin.readline())
# seq = list(map(int, stdin.readline().split()))
# stack = []
# result = [-1] * N
# c = Counter(seq)
# for n in range(N):
#     while stack and c[seq[stack[-1]]] < c[seq[n]]:
#         result[stack.pop()] = seq[n]
#     stack.append(n)
# print(*result)

# 1918 후위 표기식
# from sys import stdin
# data = list(stdin.readline().strip())
# result = ""
# stack = []
# for w in data:
#     if w.isalpha():
#         result += w
#     else:
#         if w == "(":  # 괄호가 시작함을 알려줌
#             stack.append(w)
#         elif w == "*" or w == "/":
#             while stack and (stack[-1] == "*" or stack[-1] == "/"):  # 자신과 우선순위가 같은 연산자 나올때까지 pop
#                 result += stack.pop()
#             stack.append(w)
#         elif w == "+" or w == "-":  # +-는 우선순위가 최하이므로 괄호속 연산자 모두 pop 해서 추가
#             while stack and stack[-1] != "(":
#                 result += stack.pop()
#             stack.append(w)
#         elif w == ")":  # 닫는 괄호 만나면 괄호속 연산자 모두 pop 해서 추가
#             while stack and stack[-1] != '(':
#                 result += stack.pop()
#             stack.pop()  # "("를 pop 해서 괄호가 끝났음을 의미하게 함
# while stack:  # 남은 연산자 결과에 순서대로 모두 추가하기
#     result += stack.pop()
#
# print(result)

# 1935 후위 표기식2 (미완)
# from sys import stdin
# N = int(stdin.readline())
# data = list(stdin.readline().strip())
# result =
# for n in range(N):















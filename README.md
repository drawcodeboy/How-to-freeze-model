# How to freeze model
## Question 1: How to freeze
### 1-1. Code
```
# Freeze 시키는 방법: Paramter의 gradient를 계산하지 않으면 된다.
param.requires_grad = False

# (1) 모듈을 지정하여 freeze하는 case, 2보다 1의 경우를 자주 사용
for param in model.encoder.parameters():
    param.requires_grad = False

# (2) Child를 순회하면서 모듈을 찾아 freeze 하는 case
for name, child in model.children(): # Search child module
    if name == 'encoder': # Find module to freeze
        for param in child.parameters(): # parameters in module to freeze
            param.requires_grad = False # turn off requires_grad
```
### 1-2. Freeze Module Recursively?
* 모듈 내에 여러 모듈이 더 존재하는 경우에 모두 재귀적으로 freeze하는가? -> 그렇다.
* 자세한 사항 <a href="./recursive_test.py"><code>recursive_test.py</code></a> 참고

## Question 2
* 후반 레이어가 freeze 되어도 초반 레이어는 업데이트 할 수 있는가?
* 할 수 있다. 왜냐하면, Chain Rule로 Back Propagation을 한다고 했을 때, gradient 값이 아니라 gradient를 구하는 미분 함수를 알고 있으면 후반에서 freeze 된 레이어의 경우에도 초반 레이어에 이를 전달 할 수 있다.
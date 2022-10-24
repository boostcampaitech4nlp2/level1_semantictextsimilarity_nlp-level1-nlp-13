# PR 절차
## 브렌치 종류
아래 branch 이외에 필요한 branch를 추가할 수도 있습니다.

### main branch:
 - develop branch에서 리더보드 점수나 소스코드에 큰 변화가 생겨, 다른 브렌치들도 follow up이 필요할 시 업데이트합니다.
 - 피어세션에서 업데이트하며 아래 명령어를 사용합니다. 협의 하에 rebase를 사용할 수도 있습니다.
 ```commandLine
 git checkout main
 git merge develop
 git push origin main
 ```
 - main branch가 업데이트될 때, 다른 브렌치들도 main branch를 merge해 follow up합니다. (optional)
 - 결과가 제출된 main branch는 모델과 성능지표를 포함한 tag를 붙입니다.
 
### develop branch:
 - 공동으로 사용하는 브렌치입니다. 
 - pull해서 개인 브렌치 또는 실험 브렌치로 가져가 사용합니다.
 - 개인 브렌치나 exp의 코드 작업이 성공적으로 진행되어, 다른 사람들도 사용할 필요가 있을 때 업데이트합니다. 
 - 1명 이상의 동료에게 리뷰를 받은 뒤, 직접 PR을 승인합니다.

### exp_{실험내용} branch:
- 실험이 필요한 경우나 2인 이상의 공동 작업이 임시로 필요할 때 만들어 활용합니다.

### prevate branch:
- 각자의 이름으로 설정한 브렌치입니다. 다른 사람이 수정할 수 없습니다.

## tag하기
- **나중에 참고할 수도 있는 중요한 버전**은 tag를 붙입니다.
아래는 예시입니다.
```commandLine
# git tag -a {tag 이름} -m "{설명}"
git tag -a main_ROBERT_PCC90 -m "robert, pcc 90, burketing 사용"
```

## 세팅하기
~~1. 공통 저장소를 자신의 계정으로 fork 받고, fork된 개인 원격 저장소를 clone 한다.~~ **-> fork는 사용 불가**
1. 원격 저장소를 git clone 합니다.
2. 각자의 이름으로 브랜치를 생성합니다. ex) git branch sglee   
(forked repo 대신 개인 원격저장소 대용으로 사용합니다.)

## PR하기
1. 코드를 로컬 저장소의 개인 브렌치에서 수정합니다.
2. 수정이 성공적으로 완료될 경우, add + commit후, 개인 브렌치를 push 합니다.
3. 다른 사람들도 써야하는 코드 블럭이 완성될 경우, 목적에 따라 develop branch 또는 exp branch에 PR을 합니다.
4. 왼쪽의 base repository가 공통 저장소의 main이고, 오른쪽의 compare대상이 PR을 보낼 브랜치가 맞는지 잘 확인합니다.
5. PR 제목과 commit 메세지를 적절하게 설정합니다.
6. 동료의 피드백을 받고, PR을 보낸 본인이 직접 merge를 합니다.
  - PR된 상태 : 코드리뷰 진행중인 상태
  - merge된 상태 - 코드리뷰가 완료된 상태
7. 실수로 잘못 머지했을 경우 Revert기능을 이용하여 PR을 보냅니다.

# 참고 자료
[우아한 형제 git flow](https://techblog.woowahan.com/2553/)

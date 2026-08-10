# FR3 블록 밀기 환경

## 목적과 책임 범위

이 프로젝트는 기존 OMY-to-FR3 텔레오퍼레이션 컨트롤러로 조작하는 MuJoCo 전용 평면 블록 밀기 태스크를 추가한다.

OMY MuJoCo 순기구학(FK), Cartesian 매핑, 누적 클러치 동작, 목표값 조정, velocity-DLS 역기구학(IK), null-space 자세 제어, FR3 명령 및 FR3 에셋의 기준 구현은 계속 `OMY_FRANKA_TELEOP`에 있다. 이 프로젝트는 테이블, 블록, 고정 목표 영역, 카메라, 기본 밀기 도구, 리셋, 태스크 평가, 뷰어, 최종 시뮬레이션 루프를 담당한다. 의존성은 단방향이다. `dp-act-policy-study`가 텔레오퍼레이션 구현을 가져오며, 텔레오퍼레이션 저장소는 이 태스크를 알지 못한다.

실행기는 하나의 최종 FR3/태스크 `MjModel`, 하나의 `MjData`, 하나의 `mj_step` 루프를 소유한다. 별도의 OMY 모델은 기존 텔레오퍼레이션 구현과 마찬가지로 리더 FK 계산에만 사용하며, 두 번째 시뮬레이터로 step하지 않는다.

## 의존성 설정

의존성 루트는 다음 우선순위에 따라 한 번만 결정한다.

1. `--teleop-root`
2. `OMY_FRANKA_TELEOP_ROOT`
3. 인접한 `../OMY_FRANKA_TELEOP` 디렉터리(소문자 이름도 허용)

의존성 저장소에는 `launch/FR3_omy_bridge.py`, `mujoco_menagerie/franka_fr3/fr3.xml`, OMY 모델이 있어야 한다. 두 Python 환경 모두 `pyzmq`가 필요하다(`27.1.0`에서 검증).

```bash
export OMY_FRANKA_TELEOP_ROOT=../OMY_FRANKA_TELEOP
```

## 환경 확인 및 실행

장면 확인에는 ROS나 OMY 하드웨어가 필요하지 않다.

```bash
python projects/fr3_block_push/scripts/inspect_scene.py \
  --teleop-root ../OMY_FRANKA_TELEOP
```

연구 저장소 루트에서 OMY 리더와 시뮬레이션을 함께 실행한다.

```bash
./run
```

시리얼 포트가 `/dev/ttyUSB0`이 아니면 첫 번째 인자로 전달한다. 예: `./run /dev/ttyUSB1`.

키보드 조작:

- `R`: 결정론적 리셋
- `V`: 자유 시점과 고정 조작자 카메라 전환
- `SPACE`: 태스크 상태 출력
- `Q` 또는 `ESC`: 종료

실행기는 블록 위치/속도, 목표 오차, 성공 유지 시간, 태스크 상태, 텔레오퍼레이션 모드, 선택된 입력의 최신성, 리셋 횟수를 5Hz로 출력한다.

테스트는 Python 표준 라이브러리만 사용하며 ROS가 필요하지 않다.

```bash
python -m unittest discover -s projects/fr3_block_push/tests -v
```

## 프로세스 분리형 OMY 입력

기본 직접 `ros` 입력 백엔드도 사용할 수 있다. 이 장비에서 데이터를 수집할 때는 ROS 2 Humble을 Python 3.10에, LeRobot/MuJoCo를 Python 3.13에 유지하기 위해 `--input-backend zmq`를 사용한다.

```text
프로세스 A (Python 3.10)
/leader/joint_states -> 이름 재정렬/검증 -> ZeroMQ PUB
                                      tcp://127.0.0.1:5557
프로세스 B (Python 3.13)
ZeroMQ SUB -> 기존 OMY FK/매핑/조정/IK -> 최종 FR3 태스크 시뮬레이션
           -> 20Hz LeRobot 레코더
```

프로세스 A는 `rclpy`, `sensor_msgs`, `pyzmq`를 가져오지만 MuJoCo, LeRobot, FR3 컨트롤러는 가져오지 않는다. ROS 관절 `joint1`부터 `joint6`과 트리거 `rh_r1_joint`가 각각 정확히 하나씩 있는 메시지만 허용하며, 무관한 추가 관절은 허용한다. 여섯 팔 관절은 정규 순서의 MuJoCo/전송 이름 `Joint1`부터 `Joint6`으로 변환한다. 잘못된 ROS 메시지는 제한된 빈도의 경고와 함께 버린다.

프로세스 B는 receive high-water mark 1과 `CONFLATE`를 설정한 비차단 ZeroMQ `SUB` 소켓을 사용한 뒤, 즉시 읽을 수 있는 메시지를 모두 비운다. 따라서 제어 루프에 메시지가 쌓이지 않으며 가장 최신의 유효 sequence를 선택한다. ZMQ 모드에서는 외부 브리지를 컨트롤러 부분만 파싱한다. ROS import와 `OmyPose` Node 클래스는 제외하지만 기존 FK 보조 함수, Cartesian 매핑, 목표 조정, DLS IK 함수는 변경 없이 실행한다. 수집 프로세스는 `rclpy`나 `sensor_msgs`를 가져오지 않는다.

단일 파트 UTF-8 JSON 메시지의 필드는 정확히 다음과 같다.

| 필드 | 자료형 | 의미 |
| --- | --- | --- |
| `protocol_version` | 0 이상의 정수 | 현재 버전 `2` |
| `sequence` | 0 이상의 정수 | 엄격하게 증가하는 publisher sequence |
| `source_timestamp_ns` | 0 이상의 정수 | ROS `JointState.header.stamp` |
| `sender_monotonic_ns` | 0 이상의 정수 | publisher monotonic clock |
| `joint_names` | 문자열 6개 | 정규 이름 `Joint1` ... `Joint6` |
| `position` | 유한한 숫자 6개 | OMY 관절 위치, 단위 rad |
| `trigger_position` | 유한한 숫자 | 물리 `rh_r1_joint` 클러치 위치, 단위 rad |

잘못된 JSON, 누락되거나 추가된 프로토콜 필드, 잘못된 버전, 필수 관절/트리거 누락 또는 중복, 6차원이 아닌 position, NaN/Inf, 증가하지 않는 sequence는 무시한다. 최신성은 송신자 timestamp를 신뢰하지 않고 로컬 수신 monotonic time으로 판단한다. 기본 stale timeout은 `0.2 s`이다.

첫 유효 상태를 받기 전이나 입력이 stale인 동안에는 새 목표 업데이트를 비활성화하고 목표 속도를 0으로 만들며 마지막 FR3 관절 위치 목표를 유지한다. 입력이 stale인 동안 실행기는 카메라/상태/action 캡처를 중단하므로 데이터셋 frame이 추가되지 않는다. 첫 유효 상태 수신 시점과 stale 복구 직후에는 기존 클러치 경로가 새로 받은 상태에 OMY를, 현재 명령에 FR3를 고정한 뒤 delta를 적용한다. 이 방식으로 누적 움직임과 재연결 시 목표값 점프를 막는다.

ROS 2 Humble 터미널에서 프로세스 A를 시작한다.

```bash
source /opt/ros/humble/setup.bash
source /home/chan/omy_franka_teleop/open_manipulator_omy/install/setup.bash
cd /home/chan/dp-act-policy-study

/usr/bin/python3 projects/fr3_block_push/scripts/publish_omy_joint_state.py \
  --topic /leader/joint_states \
  --endpoint tcp://127.0.0.1:5557
```

선택적으로 Python 3.13 환경에서 전송 상태를 확인할 수 있다.

```bash
python3 projects/fr3_block_push/scripts/inspect_omy_ipc.py \
  --endpoint tcp://127.0.0.1:5557 \
  --stale-timeout 0.2
```

## LeRobot 시연 데이터 기록

수집 경로는 책임을 분리한다. `TeleopRunner`는 하나의 최종 MuJoCo model/data, 기존 OMY 컨트롤러, 카메라 렌더링, step 실행, 태스크 평가, 리셋을 담당한다. `FrameExtractor`는 동기화된 실행기 표본 하나를 아래 schema로 변환한다. `DatasetRecorder`만 `LeRobotDataset.create`, `add_frame`, `save_episode`, `clear_episode_buffer`, `finalize`를 담당한다. LeRobot 소스 파일은 수정하지 않는다.

### 데이터셋 스키마

데이터셋 FPS는 20Hz로 고정하며 태스크 문장은 정확히 `Push the block into the goal region.`이다. LeRobot이 `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`를 자동으로 생성한다.

| feature | `add_frame` 표현 | 의미 |
| --- | --- | --- |
| `observation.images.top` | `(96, 96, 3)` HWC `uint8`, video | `policy_camera_top`의 offscreen RGB |
| `observation.state` | `(7,)` `float32` | `fr3_joint1` ... `fr3_joint7`의 현재 `qpos`, 단위 rad |
| `action` | `(7,)` `float32` | FR3 팔 actuator 7개에 실제 기록한 관절 위치 기준값, 단위 rad |

state에서는 의도적으로 gripper, 블록 pose, 목표 pose, 태스크 성공 여부를 제외한다. 태스크 전용 값은 환경 종료 로직에만 사용하며 policy 입력으로 사용하지 않는다.

FR3 Menagerie 모델은 `fr3_joint1`부터 `fr3_joint7`을 MuJoCo `<position>` actuator로 정의한다. 실행 시 검증에서는 각 actuator가 unit-gear joint transmission이며, 고정된 양의 gain과 `-gain`의 affine position bias를 갖는지도 확인한다. 외부 컨트롤러가 `backend.hold_q_target`을 계산하면 `TeleopBackend.apply_controller`가 이를 `data.ctrl[backend.fr3_actuator_indices]`에 기록한다. 레코더는 기록 직후 적용된 `data.ctrl` slice를 읽는다. 이 값은 rad 단위의 목표 위치이며 torque, velocity, Cartesian pose, 측정 관절 상태가 아니다. gripper control은 기록하지 않는다.

### 1단계 하이브리드 ACT 실험

수집기는 `observation.ee_pose`(`qw,qx,qy,qz,x,y,z`)와 `observation.goal_distance`(m 단위 XY 거리)도 저장한다. ACT는 정확히 `observation.state` feature를 사용하므로 `projects/fr3_block_push/scripts/prepare_hybrid_dataset.py`가 이 값들을 15차원 state(관절 7개 + EE pose 7개 + XY 거리)로 묶고, 7차원 관절 위치 action은 유지한다. 원본 데이터셋은 변경하지 않으며 변환된 데이터셋은 `datasets/data_success_50_v3_hybrid`이다.

변환 후 `./train_hybrid`로 CUDA 학습을 시작한다. 기본값은 5,000 step, batch size 8, chunk size 64, 예측 chunk당 실행 action 32개다. 필요하면 `STEPS`, `BATCH_SIZE`, `CHUNK_SIZE`, `N_ACTION_STEPS`로 덮어쓴다.

## ACT/DP 실험 현황

첫 ACT 실험은 50 episode를 사용했다. policy가 목표 앞에서 반복적으로 멈췄으며, 이는 simulator/actuator 고장보다 접근 방향 및 복구 시연 부족을 가리켰다. 실패 분석과 그에 따른 데이터 수집 결론은 [기존 50개 데이터 기반 ACT 실패 분석](docs/act_original_50_failure_analysis.md)에 정리했다.

![초기 진동 및 실패 rollout](../../vibration.gif)

균형 데이터셋은 기존 50 episode에 오른쪽 접근 30개, 중앙 접근 30개, 왼쪽 접근 30개를 더한 총 140 episode다. 같은 고정 초기조건과 5,000 학습 step에서 검증한 결과는 다음과 같다.

| 지표 | ACT | Diffusion Policy |
| --- | ---: | ---: |
| 데이터셋 | balanced-140 | balanced-140 |
| 학습 step | 5,000 | 5,000 |
| 평가 episode | 5 | 5 |
| 성공 episode | 5/5 | 1/5 |
| 성공률 | 100% | 20% |
| 최종 목표 거리 평균 | 0.0312m | 0.0778m |
| 최대 측면 이탈 평균 | 0.0270m | 0.0795m |

### ACT balanced-140 성공 롤아웃

다음 압축 영상은 균형 잡힌 140-episode 데이터셋으로 학습한 ACT policy가 블록 밀기 태스크를 완료하는 모습을 보여준다.

![ACT balanced-140 성공 롤아웃](../../results/ACT_sucess_data_140.gif)

전체 비교표, checkpoint/config 검증, episode별 지표, DP 학습 step ablation 계획은 [ACT와 DP의 동일 5k 학습 예산 비교](docs/act_vs_dp_5k_comparison.md)에 정리했다.

관측/action 정렬은 integration 이전 시점 기준이다. 시간 `t`에서 기존 컨트롤러가 먼저 `action_t`를 계산하고 기록한다. 실행기는 `image_t`, 현재 `qpos_t`, 적용된 `action_t`를 캡처한 다음 `mj_step`을 호출한다. 표본 시점은 wall time이 아니라 `data.time`을 사용해 `0.05 s` 간격으로 결정한다. LeRobot은 frame index와 선언된 20 FPS로 timestamp를 생성한다. terminal event에서는 정규 간격을 벗어난 frame을 추가하지 않는다.

`policy_camera_top`은 최종 태스크 `MjModel`/`MjData`에서 `mujoco.Renderer`로 렌더링하며, 대화형 viewer framebuffer는 캡처하지 않는다. 장면 검사 결과 HWC `uint8`을 직접 반환했다. 128px 해상도에서 블록을 world `y=-0.2`에서 `y=+0.2`로 옮기면 이미지 row 89에서 row 38로 이동해 renderer의 일반적인 좌상단 원점을 확인했다. 따라서 수집 과정에서 이미지를 상하 반전하지 않는다. MuJoCo가 RGB를 반환하므로 BGR 변환도 적용하지 않는다. 저장된 video를 LeRobot으로 불러오면 기본 decoder가 channel-first 부동소수점 tensor로 보여줄 수 있다. 이 읽기 시점 표현은 검증된 HWC `uint8` 기록 경계나 feature metadata를 바꾸지 않는다.

### 수집 생명주기와 조작키

연구 저장소 루트에서 세 수집 프로세스를 모두 실행한다. wrapper는 ROS bridge를 Python 3.10 subshell에 유지하고 수집기는 현재 Python 3.12+ 환경에서 실행한다.

```bash
./data \
  --dataset-root /home/chan/dp-act-policy-study/datasets/fr3_block_push_run_001 \
  --episodes 30
```

인자 없이 실행하면 `datasets/` 아래의 새 timestamp 디렉터리에 저장된 episode 30개를 수집한다. 필요하면 첫 번째 인자로 시리얼 포트를 전달한다. 예: `./data /dev/ttyUSB1 --episodes 30`. `python3`가 원하는 LeRobot 환경이 아니면 `FR3_DATA_PYTHON`으로 수집기 interpreter를 선택할 수 있다.

기본 로컬 루트는 `datasets/fr3_block_push`다. 선택한 루트가 이미 존재하면 실행을 거부한다. 기록 과정은 기존 데이터셋에 append하거나 overwrite하거나 upload하지 않는다. `--repo-id`는 metadata일 뿐이며 기본값은 `local/fr3_block_push`다. Hub push는 없다.

조작키는 기존 `run_teleop.py` binding과 충돌하지 않는다(`R`, SPACE, `Q`/ESC의 의미는 유지).

- `C`: 완전히 리셋한 뒤 새 episode 기록 시작
- `S`: 수동 중지 또는 성공 여부와 관계없이 현재 buffer의 episode 저장
- `D`: 현재 buffer를 버리고 리셋한 뒤 즉시 다시 기록
- `R`: 현재 buffer를 버리고 FR3 home으로 리셋한 뒤 대기 상태로 전환
- `V`: 조작자 viewer의 자유 카메라와 고정 전면 시점 전환
- `SPACE`: 수집 및 태스크 상태 출력
- `Q` 또는 `ESC`: 저장하지 않은 buffer를 버리고 데이터셋을 finalize한 뒤 종료

episode는 리셋이 끝난 뒤에만 시작한다. 성공, timeout, 작업 영역 이탈 시 sampling을 중지하고 시뮬레이션 화면을 유지한다. 자동으로 저장하지 않으므로 `S`로 승인하거나 `D`로 버리고 다시 기록한다. terminal 상태가 아닌 현재 episode도 `S`로 저장할 수 있다. 빈 episode는 거부한다. 모든 reset 경로는 다음 episode 시작 전에 현재 LeRobot buffer를 비우므로 reset 전후 frame이 섞이지 않는다.

완성된 로컬 데이터셋은 다음 명령으로 검사한다.

```bash
python projects/fr3_block_push/scripts/inspect_recorded_dataset.py \
  /home/chan/dp-act-policy-study/datasets/fr3_block_push_run_001 \
  --repo-id local/fr3_block_push
```

이 명령은 episode/frame 수, FPS, 모든 feature, 첫 표본의 image/state/action/timestamp/episode/frame shape와 dtype을 출력한다. 학습이나 변환은 수행하지 않는다.

## 장면 구성

실행 시 `scene_builder.py`가 외부 FR3 XML을 파싱하고 mesh 디렉터리를 찾은 뒤 로컬 태스크 요소를 병합하고, 기존 `fr3_hand` frame 아래에 로컬 기본 밀기 도구를 삽입한다. 그 결과로 만든 메모리 내 XML을 한 번 compile한다. FR3 XML이나 컨트롤러 구현을 복사하지 않으며, 절대 로컬 경로를 commit하지 않고, 생성 artifact도 저장하지 않는다.

설치된 MuJoCo 버전에서는 절대 경로 XML include가 포함된 FR3 파일의 상대 mesh 디렉터리를 보존하지 않기 때문에 실행 시 XML을 결합하는 방식을 사용한다. builder는 두 소스 저장소를 모두 변경하지 않는다.

## 태스크 정의

고정 테이블 상단은 `z=0.20 m`다. half-size `0.035 m`, 질량 `0.30 kg`인 free-joint 블록은 `(0.62, 0.00, 0.235)`에서 시작한다. 충돌하지 않는 시각적 목표 영역은 `(0.80, 0.00, 0.202)`를 중심으로 하며 평면 half-size는 `0.075 m`다. 상단 policy 카메라는 작업 영역 위에 고정한다. 별도 overhead 카메라는 조작자 viewer preset일 뿐이며 기록하지 않는다. `V`로 이 preset과 viewer 자유 카메라를 전환한다. FR3 hand에 부착한 capsule pusher의 끝은 home configuration에서 테이블 바로 위에 놓인다.

리셋은 외부 FR3 `home` keyframe에서 일곱 FR3 관절과 actuator 목표를 복원하고, 사용하지 않는 gripper 명령을 닫으며, 블록 free-joint pose를 복원한다. 모든 velocity와 actuator state, 태스크 timer를 초기화한 뒤 `mj_forward`를 호출한다. 기본 리셋은 결정론적이며 `BlockPushConfig`에서 seed 기반의 작은 XY 무작위화를 활성화할 수 있다.

성공하려면 margin을 포함한 블록 전체가 목표 안에 들어간 상태에서 평면 속도 `0.01 m/s` 이하를 `0.5 s` 동안 유지해야 한다. 설정된 테이블 작업 영역을 벗어나거나 episode timeout `120 s`에 도달하면 실패다. 평가는 상태를 읽기만 하며 제어 루프를 변경하지 않는다.

## 참고 자료와 한계

제공된 `Push_MuJoCo` 파일은 일반적인 테이블, free block, 시각적 목표, 카메라, pusher, reset, substep 아이디어를 조사하는 용도로만 사용했다. 해당 README는 <https://github.com/JericLew/Push_MuJoCo>와 MuJoCo Menagerie 출처를 명시하지만 프로젝트 license는 명시하지 않는다. 따라서 원본은 `references/Push_MuJoCo/` 아래에 로컬 전용으로 두고 Git에서 제외했다. production code는 이를 import하거나 include하지 않으며 Panda 의존성도 없다.

이 환경은 simulation 전용이며 실제 로봇 안전성을 보장하지 않는다. 환경 구현 자체에는 ACT 또는 Diffusion Policy 학습, Hub upload, multi-camera 데이터, multi-goal 태스크, gripper policy, 데이터셋 resume/append가 포함되지 않는다.

ROS 2 Humble의 Python 3.10과 LeRobot의 Python 3.13 ABI 충돌은 ZMQ 프로세스 경계로 분리했다. publisher를 재시작하면 sequence counter가 초기화되며, 현재의 엄격한 anti-regression 정책에서는 수집기도 함께 재시작해야 한다. 실제 하드웨어의 ROS 관절 이름/주기, 장시간 publisher 복구, 실시간 기록 중 viewer key, 실제 OMY로 수행한 성공 시연은 아직 검증하지 않았다.

마지막으로 action은 high-gain MuJoCo position-servo 기준값이다. 접촉, saturation, 빠른 움직임 중에는 측정 `qpos`와 다를 수 있다. 이를 torque로 해석해서는 안 되며, sim-to-real 적용에는 별도로 검증한 controller/action interface가 필요하다.

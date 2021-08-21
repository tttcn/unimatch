# UniMatch

UniMatch

UniMatch 是一个基于视觉地图的定位算法。

UniMatch is a positioning algorithm based on visual map.

本仓库包含以下内容：

1. 一个 UniMatch 算法的 [python 实现](unimatch/core.py)。
2. 一个用于调试的 [web 界面](unimatch/gateway.py)。
3. [关于项目的 wiki](https://github.com/tttcn/unimatch/wiki) - wiki里包含了项目相关的详细信息（不定期更新）。
4. 构建 UniMatch 算法所需地图的相关代码。

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [使用说明](#使用说明)
    - [项目结构](#项目结构)
    - [启动服务](#启动服务)
- [示例](#示例)
- [相关仓库](#相关仓库)
- [维护者](#维护者)
- [如何贡献](#如何贡献)
- [使用许可](#使用许可)

## 背景

`UniMatch` 是为了实现机场中多传感器融合定位系统而研发的一套基于视觉地图的定位算法。算法通过 HNSW 结构的视觉地图来提高定位效率和准确度，有良好的全局精度和鲁棒性。

本项目为算法的 python 实现，使算法可以通过一个 http 请求简单调用，方集成到多传感器融合方案中。

这个仓库的目标是：

1. 一套可以高效运行的视觉地图定位算法。
2. 一套关于本项目如何使用和开发的完整文档。
3. 一个简单的地图测试和项目调试 web 界面。
4. 一个交流和维护的平台，如果有希望添加的功能和 bug 修复请求，请通过提交 issue 完成，后来者可以从历史 issue 中获得解决方案。
5. TODO：整理生成地图相关的代码 & web 操作界面。

## 安装

这个项目使用 [poetry](https://python-poetry.org/) 进行开发所需 python 环境的部署，如果想要进行开发，请按照官方网站的提示安装。也提供了 [docker](https://www.docker.com/) 部署的方式。

```sh
# 此时应该在项目的根目录下
$ poetry install
```

## 使用说明

### 项目结构

项目的目录结构如下：
```sh
.
├── dockerfile
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
└── unimatch
    ├── config
    │   └── default.ini
    ├── core.py
    ├── database
    │   ├── db_index.bin
    │   ├── db_index.bin.dat
    │   ├── db_features.bin
    │   └── db_map_poses.bin
    ├── gateway.py
    ├── __init__.py
    └──  map.py
```

为了算法能够正常运行，需要在 config 目录下放置运行时需要的配置文件，为python标准配置文件格式，项目中为默认的配置文件。此外，在 database 目录下放置描述视觉地图的二进制文件，定位算法在初始化时将会读取这些文件。
> 在使用 docker 镜像时应该将 config 和 database 两个目录进行映射，并且提前在目录中放置需要的配置文件和数据文件。目前 database 目录下的4个文件名都是写死在代码中的，必须和目录结构中所示名称一致。

### 启动服务

在开发中启动算法服务的方式很简单，使用 poetry 安装好开发环境之后可以用一行命令启动服务，默认端口为8000，可以通过 http://localhost:8000/ 访问，接口用法可见[ wiki 页面](https://github.com/tttcn/unimatch/wiki)。

```sh
$ cd unimatch
# 此时应该在 ./unimatch 目录下
$ poetry run uvicorn gateway:app --reload
```

在 docker 中已经配置了相应的启动命令，只需要注意将 config 和 database 两个目录进行映射，并且提前在目录中放置需要的文件。

```sh
docker run -d -p 8000:8000 --name unimatch_service -v config文件夹的绝对路径:/code/unimatch/config -v database文件夹的绝对路径:/code/unimatch/database unimatchdocker:buster
```

如果出现问题可以检查容器的 log 和各项配置属性，可以自查的常见问题有：

- 目录映射是否正确。
- 目录中是否放置了需要的 config 文件和4份 database 文件，文件名是否正确。


## 示例

想了解项目运行所需要的地图应该如何布置，对应的配置应该如何写，请参考[如何制作地图](https://github.com/tttcn/unimatch/wiki/%E5%BB%BA%E5%9B%BE)。

## 相关仓库

如果想要对本项目进行开发和改造，建议了解以下的相关项目：

- [nmslib](https://github.com/nmslib/nmslib) — 项目中使用的 HNSW 算法的实现。
- [OpenCV](https://opencv.org/) — 项目中使用了许多 OpenCV 的算法。
- [FastAPI](https://fastapi.tiangolo.com/) — 项目中使用 FastAPI 作为网关。

## 维护者

[@tttcn](https://github.com/tttcn)。

## 如何贡献

非常欢迎你的加入！如果你发现了 bug，或者对项目有任何好的提议，[提一个 Issue](https://github.com/tttcn/unimatch/issues/new)或者提交一个 Pull Request。

### 贡献者

感谢以下对项目有所帮助的贡献者（排名不分先后）：
王运涛，王蕾，李梦琪，杨辞源，许书畅，王丰，潘泽文，庄煜洲，逄立飞。

## 使用许可

[MIT](LICENSE) © Taotao Tang
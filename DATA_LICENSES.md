# 数据来源与再分发说明

三个核心数据来源的存档许可均为 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。许可属于各数据集，不能据此认定第三方模型代码、权重或文献也采用相同许可。

## OP

数据提供方：UK Power Networks，Optimise Prime，WS2 Table 27 depot load data。

- [官方目录](https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-optimise-prime/information/)
- 原文件：`depot_load.csv`，74,421,688字节。
- SHA256：`db90bec45b374d93792943c0755e0dcd88ca91e5991ad974df91f9a37efbac43`。
- raw包内原CSV字节未改动。衍生输入采用每个整点前300秒内最后一条唯一时间戳的非负有限功率，边界规则为 `(g-300s, g]`。零值和缺失分别保留。
- 未发布站点身份调查材料或额外身份映射。

## AMS

Sørensen, Åse Lekang (2021). Residential electric vehicle charging datasets from apartment buildings. Mendeley Data, V1. DOI：[10.17632/jbks2rcwyj.1](https://doi.org/10.17632/jbks2rcwyj.1)。

- 原文件：`Dataset 5_AMS data from garage Bl2.csv`，500,329字节。
- SHA256：`8e4d0d0deae085dd21f148d1930ce57e55e392dc89f603ba82f076b06799729e`。
- raw包内原CSV字节未改动。只使用时间和AMS_kWh进行研究；源文件中的合成功率情景列不作为实测输入。
- 衍生数据沿用本地民用时钟，排除不合格时间窗口，不填补缺失电量。

## LCL

数据提供方：UK Power Networks / London Datastore，Low Carbon London Electric Vehicle Load Profiles。

- [官方目录](https://data.london.gov.uk/dataset/low-carbon-london-electric-vehicle-load-profiles-2y5k8)
- [原始完整发布包](https://data.london.gov.uk/download/2y5k8/c869f8ac-33d3-4cf9-bf28-d89c61bbc7a2/EV%20Load%20Profiles.zip)
- 原始完整ZIP SHA256：`13bbb60211f45bd19c65d8ec5f1c81032ca068f169e43fb4ebec446df5481736`。
- 本仓库不镜像整个ZIP。raw包包含固定48个居民计量对象的ReadingDate与Active Power Total (Watts)两列，保留源字符串，选取 `2013-02-01 <= timestamp < 2014-02-01`。CSV格式重新写出，各投影文件的SHA256见清单。
- 未收录问卷、身份映射、商业和公共试验工作簿，也未收录后续未使用时期。投影文件是经过字段与时期筛选的衍生材料，不冒充原始ZIP。
- 目标为来源整点平均功率记录；物理单位换算尚未确认，模型误差仍使用source_raw。

## 代码与引用

TimeXer固定代码版本为 `4e938a1767106324dd753b2a44832bf870a0252e`。使用的5个代码/许可文件位于 `repro/vendor/`，保留上游MIT许可证和THUML署名。不是把TimeXer重新命名为原创模型。

其他依赖由固定版本的requirements安装，不镜像依赖环境或预训练模型权重。数据发布方不因本次引用和再分发而认可本研究结论。

<img align="right" width="200" height="200" src="https://devplatform.tcsbank.ru/ml-core/notebooks/static/MLCORE.png">

<h1 style='margin-left:5%;'> Ноутбуки </h1>

<p style='margin-left:5%;'> Добро пожаловать на Welcome страничку о ноутбуках! </p>

<p style='margin-left:5%;'><b>С полной информацией можно ознакомиться на главной странице о ноутбуках:<a href="https://ai-platform.pages.devplatform.tcsbank.ru/user-docs/docs/notebooks/getting_started/"> Notebooks</a></b></p>.

<section style='text-align:center; margin-top:70px'>

## Структура ноутбука
<p style="width:70%; margin-left:15%">
    Несмотря на название, созданный ноутбук - это пространство, где можно создавать <b>.ipynb</b> и другие файлы. В случае, если нужно работать с большим количеством файлов - рекомендуем подключить шару, либо воспользоваться <b>s3</b>.
</p>

<h4 style="margin-left:15%; text-align:left;">Возможности и ограничения ноутбуков</h4>
    
<ul style="margin-left:15%; text-align:left;">
    <li> Для каждого ноутбука можно выбрать один из флейворов -- набора ресурсов, которые будут выделяться при его запуске. Со списком доступных флейворов можно ознакомиться в полной документации. </li>
    <li> В случае необходимости можно поменять название, примонтированную шару, флейвор и описание -- все это делается на странице со <a href='https://devplatform.tcsbank.ru/ml-core/notebooks/'> списком ноутбуков </a> в настройках нужного ноутбука. </li>
    <li> У внутреннего хранилища ноутбука есть предел - <b>4gb для личных ноутбуков и 8gb для командных</b>.
</ul>


## У меня что-то сломалось
<p style="width:70%; margin-left:15%">
    По любым вопросам и пожеланиям смело обращайтесь в канал $#ml-core-ask$ :)
</p>    
</section>


</br>

</br>

</br>

---

<h1 style='margin-left:5%; margin-right:5%;'> Плагины </h1>

<p style='margin-left:5%; margin-right:5%;'> В MLC Ноутбуки установлены плагины, полный список можно увидеть на все той же страничке Notebooks (ссылка в самом верху). Тут же рассмотрим основные из них. </p>

<section style='text-align:center; margin-top:70px'>

## Gitlab Extension
<p style="width:70%; margin-left:15%">
    Как уже понятно из названия, теперь можно какждый ноутбук сделать репозиторием у себя (или у команды) в гитлабе. Более того, <code>commit</code>, <code>push</code> и <code>pull</code> можно делать прямо из ноутбука. Все нужные инструменты можно найти на панели слева (нажав на значок гита), а также в верхнем меню, в поле <code>Git</code>.
    <details style='margin-left:15%; margin-right:15%; text-align:left;'>
    <summary> Подробнее про подключение репозитория и работу с ним</summary>
    <div>
    Существует 2 основных сценария: 
    </div>
    <ol>
        <li> Сначала клонируем репозиторий в ноут, работаем с ним, потом пулим.
            <div>Для этого: в созданном ноутбуке жмем на поле <code>Git</code> в верхнем меню (Там же, где <code>File</code>, <code>Edit</code>) &#8594 Clone a Repository &#8594 Вставляем ссылку на .git &#8594 При необходимости авторизуемся черех учетку гита.</div>
        <li> Создаем репозиторий из ноута, инициализируем его на гите, а дальше работаем как обычно.
            <div>Для этого: <code>Git</code> в верхней панельке &#8594 Initialize a Repository &#8594 Соглашаемся &#8594 снова <code>Git</code> &#8594 Add Remote Repository &#8594 вставляем ссылку на гит, с которым хотим связать &#8594 авторизуемся при необходимости.</div>
    </ol>
    <div>
        <b>Как коммитить</b> - жмем на значок гита в левой панельке -> выбираем нужную ветку -> в Untracked добавляем все нужное через значок + -> внизу пишем Summary и Description (если хотим) -> Commit
    </div>
    <div>
        <b>Как пушить</b> - жмем на значок гита в левой панельке -> жмем на значок облачка со стрелкой вверх
    </div>
    <div>
        <b>Как пулить</b> - жмем на значок гита в левой панельке -> жмем на значок облачка со стрелкой вниз
    </div>
    
</details>
</p>

## CPU, Mem, Disk usage

<details style="width:70%; margin-left:15%; text-align:left;">
    <div><i>CPU</i> и <i>Mem</i> отвечают за отображение потребляемых в данный момент ресурсов - если переполнение <i>CPU</i> повлечет за собой лишь замедление расчетов, то переполнение <i>Mem</i> повлечет перезапуск <code>kernel'a</code>, так что <b>будьте бдительны и внимательно следите за этим показателем</b>.
    </div>
    <div><i>Disk usage</i> показывает занятое ноутбуком место на шаре. Не забывайте об ограничениях - если шкала дойдет до конца, следующий запуск ноутбука может стать проблематичным. Если это произойдет - пишите нам в <code>#ml-core-ask</code> и мы вам поможем (но лучше чтобы не происходило).
    </div>
</details>

## Autocomplete и Autoformat
<details style="width:70%; margin-left:15%; text-align:left;">
    <div>
    Первый работает автоматически и предлагает вам дописать функцию или переменную. 
    </div>
    <div>
    Второй добавляет сверху рабочей зоны кнопку, которая отформатирует код во всех ячейках к единому формату.
    </div>
</details>

## Snippets
<p style="width:70%; margin-left:15%; text-align:left;">
    Плагин с полезными заготовками кода, которые можно использовать в своих ноутбуках.
    Внутри него вы найдете различные сниппеты для работы - от подключения к DWH до работы c ClearML.
    Плагин находится на правой панели - с иконкой `</>`
</p>

</section>

</br>

</br>

</br>

---

<h1 style='margin-left:5%; margin-right:5%;'> Полезные библиотеки и советы </h1>

<p style='margin-left:5%; margin-right:5%;'> В решении стандартных задач могут помочь следующие советы: </p>

<section style='text-align:center; margin-top:30px'>
<p style='margin-left:15%; text-align: left;'><b> Как подключиться к s3? </b></p>
<details style='margin-left:18%; margin-right:18%; text-align: left;'> 
    Первый вариант - подключить в настройках ноутбука. Второй вариант - использоваться библиотеку <code>boto3</code>.
    <div> <b>Пример кода:</b> </div>
    <div><code style="background-color: #eee; border-radius: 3px; font-family: courier, monospace; padding: 4px 0 4px 0;">import boto3
from getpass import getpass
file_name = "test.txt"
path_to_save = "model_checkpoints/" + file_name
access_key = getpass('Enter your ACCESS_KEY')
secret_key = getpass('Enter your SECRET_KEY')
s3 = boto3.client('s3',
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key,
                 endpoint_url = 'https://s3.tinkoff.ru')
bucket_name = "{BUCKET_NAME}"
s3.download_file(bucket_name, path_to_save, file_name)
s3.upload_file(file_name, bucket_name, path_to_save)
</code> 
</div>
        
</details>
<p style='margin-left:15%; margin-top:10px; text-align: left;'><b> Как вставить секреты? </b></p>
<details style='margin-left:18%; margin-right:18%; text-align: left;'> 
        Пока что хранить секреты не удастся, но используя метод <code>getpass()</code> библиотеки <code>getpass</code> (то есть <code>from getpass import getpass</code>) можно вставлять секреты через стандартный ввод, и весь ввод закрасится звездочками - так вполне можно работать с секретами в командных ноутбуках.
<div><b>Пример кода:</b></div>
<div><code style="background-color: #eee; border-radius: 3px; font-family: courier, monospace; padding: 4px 0 4px 0;">from getpass import getpass
import pandas as pd
def get(self, query):
    with psycopg2.connect(
        host="pgb-ai-platform.pgsql.tcsbank.ru",
        port="5432",
        database="ai_platform",
        user='grafana',
        password=getpass(),
    ) as con:
        df = pd.read_sql_query(query, con)
    return df
</code>
</div>
</details>

<p style='margin-left:15%; margin-top:10px; text-align: left;'><b>Как подключаться к внутренним сервисам tinkoff?</b></p>
<details style='margin-left:18%; margin-right:18%; text-align: left;'>
        С довольно большим количеством подобных проблем поможет справиться библиотека <a href="https://wiki.tcsbank.ru/pages/viewpage.action?pageId=989694851">tinkoffpy</a> (по этой же ссылке есть и примеры использования). Если же нужно подключиться к какому-то конкретному адресу -- обратитте внимание на библиотеку <code>psycopg</code>.
<div><b>Пример кода</b></div>
<div><code style="background-color: #eee; border-radius: 3px; font-family: courier, monospace; padding: 4px 0 4px 0;">from getpass import getpass
import pandas as pd
import psycopg2
def get(self, query):
    with psycopg2.connect(
        host="pgb-ai-platform.pgsql.tcsbank.ru",
        port="5432",
        database="ai_platform",
        user='grafana',
        password=getpass(),
    ) as con:
        df = pd.read_sql_query(query, con)
    return df
</code>
</div>
<div><b>Установка psycopg</b></div>
<div><code> !pip install psycopg2-binary
</code>
</div>
</details>
    

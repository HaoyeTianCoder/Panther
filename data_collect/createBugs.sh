dir=/Users/haoye.tian/Documents/University/project/defects4j_buggy/
proj=Chart
for bugId in $(seq 1 26)
do
	defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Closure
for bugId in $(seq 1 133)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Lang
for bugId in $(seq 1 65)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Math
for bugId in $(seq 1 106)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Mockito
for bugId in $(seq 1 33)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Time
for bugId in $(seq 1 27)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Jsoup
for bugId in $(seq 1 93)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=JacksonDatabind
for bugId in $(seq 1 112)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done
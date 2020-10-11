dir=/Users/haoye.tian/Documents/University/project/defects4j_buggy/

proj=Chart
for bugId in $(seq 1 26)
do
	/Users/haoye.tian/Documents/University/project/defects4j/framework/bin/defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done
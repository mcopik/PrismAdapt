/**
 * 
 */
package simulator.gpu.opencl.kernel.expression;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author mcopik
 *
 */
public class ExpressionList implements KernelComponent
{
	List<Expression> exprs = new ArrayList<>();
	boolean hasIncludes = false;
	boolean hasDeclarations = false;

	public void addExpression(Expression expr)
	{
		exprs.add(expr);
		if (!hasIncludes && expr.hasIncludes()) {
			hasIncludes = true;
		}
		if (!hasDeclarations && expr.hasDeclaration()) {
			hasDeclarations = true;
		}
	}

	public void addExpression(ExpressionList expr)
	{
		Iterator<Expression> it = expr.iterator();
		while (it.hasNext()) {
			addExpression(it.next());
		}
	}

	public Iterator<Expression> iterator()
	{
		return exprs.iterator();
	}

	@Override
	public boolean hasIncludes()
	{
		return hasIncludes;
	}

	@Override
	public boolean hasDeclaration()
	{
		return hasDeclarations;
	}

	@Override
	public List<Include> getIncludes()
	{
		if (!hasIncludes) {
			return null;
		}
		List<Include> includes = new ArrayList<>();
		for (Expression expr : exprs) {
			if (expr.hasIncludes()) {
				includes.addAll(expr.getIncludes());
			}
		}
		return includes;
	}

	@Override
	public KernelComponent getDeclaration()
	{
		if (!hasDeclarations) {
			return null;
		}
		ExpressionList declarations = new ExpressionList();
		for (Expression expr : exprs) {
			if (expr.hasDeclaration()) {
				KernelComponent decl = expr.getDeclaration();
				if (decl instanceof Expression) {
					addExpression((Expression) decl);
				} else {
					addExpression((ExpressionList) decl);
				}
			}
		}
		return declarations;
	}

	@Override
	public void accept(VisitorInterface v)
	{
		for (Expression expr : exprs) {
			v.visit(expr);
		}
	}

	@Override
	public String getSource()
	{
		StringBuilder builder = new StringBuilder();
		for (Expression expr : exprs) {
			builder.append(expr.exprString).append("\n");
		}
		return builder.toString();
	}
}

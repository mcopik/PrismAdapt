//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel.expression;

import simulator.gpu.opencl.kernel.expression.ExpressionGenerator.Operator;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.StdVariableType;

public class ForLoop extends ComplexKernelComponent
{
	private Expression definition;
	private CLVariable counter;
	private Expression endValue = null;
	boolean decreasing = false;

	public ForLoop(CLVariable counter, boolean decreasing)
	{
		this.counter = counter;
		definition = new Expression(";");
		this.decreasing = decreasing;
	}

	public ForLoop(CLVariable counter, long startValue, long endValue)
	{
		this.counter = counter;
		definition = ExpressionGenerator.createAssignment(counter, new Expression("0"));
		ExpressionGenerator.addComma(definition);
		counter.setInitValue(StdVariableType.initialize(startValue));
		this.endValue = new Expression(Long.toString(endValue));
	}

	public ForLoop(String counterName, long startValue, long endValue)
	{
		counter = new CLVariable(new StdVariableType(startValue, endValue), counterName);
		counter.setInitValue(StdVariableType.initialize(startValue));
		definition = counter.getDefinition();
		this.endValue = new Expression(Long.toString(endValue));
	}

	public ForLoop(String counterName, long startValue, long endValue, boolean decreasing)
	{
		if (decreasing) {
			counter = new CLVariable(new StdVariableType(endValue, startValue), counterName);
			counter.setInitValue(StdVariableType.initialize(endValue));
			this.endValue = new Expression(Long.toString(startValue));
			decreasing = true;
		} else {
			counter = new CLVariable(new StdVariableType(startValue, endValue), counterName);
			counter.setInitValue(StdVariableType.initialize(startValue));
			this.endValue = new Expression(Long.toString(endValue));
		}
	}

	public CLVariable getLoopCounter()
	{
		return counter;
	}

	@Override
	public boolean hasDeclaration()
	{
		return false;
	}

	@Override
	public KernelComponent getDeclaration()
	{
		return null;
	}

	@Override
	public void accept(VisitorInterface v)
	{
		for (Expression expr : variableDefinitions) {
			expr.accept(v);
		}
		for (KernelComponent component : body) {
			component.accept(v);
		}
	}

	@Override
	protected String createHeader()
	{
		StringBuilder builder = new StringBuilder("for(");
		if (endValue != null) {
			builder.append(definition);
			if (decreasing) {
				builder.append(ExpressionGenerator.createBasicExpression(counter.getSource(), Operator.GT, endValue));
			} else {
				builder.append(ExpressionGenerator.createBasicExpression(counter.getSource(), Operator.LT, endValue));
			}
		} else {
			builder.append(";");
		}
		builder.append(";");
		if (decreasing) {
			builder.append("--");
		} else {
			builder.append("++");
		}
		builder.append(counter.varName).append(")");
		return builder.toString();
	}
}